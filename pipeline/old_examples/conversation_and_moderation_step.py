"""
Conversation Classification and Moderation Step for the Ad Injection Pipeline.

Uses an LLM agent to classify the conversation category and language,
then moderates the prompt for ad-placement suitability. A 'rejected' moderation
outcome stops the pipeline via ConversationModerationRejectedException;
'cautioned' allows ads to proceed.
"""
import time
from typing import Optional

from ad_injection.agents.llm_agents import OpenAILLMAgentProvider
from ad_injection.agents.models import (
    AgentType,
    ConversationModerationInput,
    ConversationModerationOutput,
)
from ad_injection.pipeline.abstract import CriticalAbstractStep, StepContext
from ad_injection.pipeline.exceptions import (
    ConversationModerationStepFailedException,
    ConversationModerationRejectedException,
)
from ad_injection.pipeline.models import ConversationModerationResult, InitialResult
from ad_request.models import AdRequestModerationStatus
from ad_request.services.ad_request_service import AdRequestService
from common.services.constants_service import ConstantsService
from helper_functions.logger import get_logger

logger = get_logger(__name__)


class ConversationAndModerationStep(CriticalAbstractStep[ConversationModerationResult, InitialResult]):
    """
    Classifies the conversation and moderates the prompt for ad suitability.

    This is a critical step — if the LLM is unreachable the pipeline fails
    with ERROR. A 'rejected' moderation outcome raises
    ConversationModerationRejectedException which maps to REJECTED (not ERROR).
    """

    def __init__(
        self,
        force_default: bool = False,
        agent_provider: Optional[OpenAILLMAgentProvider] = None,
    ):
        """
        Initialize the step.

        Args:
            force_default: If True, always use execute_default().
            agent_provider: Optional pre-configured agent provider.
                            If None, a default one is created for the
                            CONVERSATION_MODERATION agent type.
        """
        super().__init__(force_default)
        self.constants_service = ConstantsService()
        self.agent_provider = agent_provider or OpenAILLMAgentProvider(
            agent_type=AgentType.CONVERSATION_MODERATION,
            model_name=self.constants_service.get_conversation_moderation_model(),
            temperature=1
        )
        self.ad_request_service = AdRequestService()

    # ------------------------------------------------------------------
    # Default (pass-through)
    # ------------------------------------------------------------------

    def execute_default(self, context: StepContext[InitialResult]) -> ConversationModerationResult:
        """
        Pass-through implementation that approves all prompts without LLM classification.

        Args:
            context: Pipeline context containing the initial ad request.

        Returns:
            ConversationModerationResult with generic defaults.
        """
        ad_request = context.ad_request
        logger.debug(f"Running default conversation & moderation step for ad_request {ad_request.id}")

        return ConversationModerationResult()

    # ------------------------------------------------------------------
    # Real implementation
    # ------------------------------------------------------------------

    def execute_implementation(self, context: StepContext[InitialResult]) -> ConversationModerationResult:
        """
        Classify the conversation and moderate the prompt via the LLM agent.

        Workflow:
        1. Send the prompt to the LLM and receive structured classification +
           moderation output.
        2. Persist AdRequestConversationMetadata (category + language).
        3. Persist AdRequestModeration (moderation_status + reason).
        4. If moderation_status is 'rejected', raise ConversationModerationRejectedException.
        5. Return ConversationModerationResult for downstream steps.

        Args:
            context: Pipeline context containing the initial ad request.

        Returns:
            ConversationModerationResult with classification and moderation data.

        Raises:
            ConversationModerationRejectedException: When the prompt is unsuitable for ads.
            ConversationModerationStepFailedException: On any unexpected failure.
        """
        ad_request = context.ad_request
        prompt = ad_request.message.prompt

        try:
            start = time.perf_counter()
            llm_output: ConversationModerationOutput = self.agent_provider.structured_conversation(
                structured_input=ConversationModerationInput(prompt=prompt),
                output_model=ConversationModerationOutput,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"LLM structured_conversation took {elapsed_ms:.1f}ms for ad_request {ad_request.id}")

            category_key = llm_output.category
            language_code = llm_output.language.lower()
            moderation_status = llm_output.moderation_status
            moderation_reason = llm_output.moderation_reason

            # Persist conversation metadata
            self.ad_request_service.attach_conversation_metadata(
                ad_request=ad_request,
                category_key=category_key,
                language_code=language_code,
            )

            # Persist moderation outcome
            self.ad_request_service.attach_moderation(
                ad_request=ad_request,
                moderation_status=AdRequestModerationStatus(moderation_status),
                reason=moderation_reason,
            )

            logger.info(
                f"Conversation moderation for ad_request {ad_request.id}: "
                f"category={category_key} language={language_code} status={moderation_status}"
            )

            if moderation_status == "rejected":
                raise ConversationModerationRejectedException(
                    f"Prompt rejected by moderation for ad_request {ad_request.id}: {moderation_reason}"
                )

            return ConversationModerationResult()

        except ConversationModerationRejectedException:
            raise
        except Exception as e:
            logger.error(f"Conversation & moderation step failed for ad_request {ad_request.id}: {e}")
            raise ConversationModerationStepFailedException(
                f"Conversation & moderation step failed: {e}"
            ) from e

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from typing import Tuple

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from agenteval import jinja_env
from agenteval.evaluators import BaseEvaluator
from agenteval.evaluators.bedrock_request.bedrock_request_handler import (
    BedrockRequestHandler,
)
from agenteval.test import TestResult

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE_ROOT = "evaluators/canonical"
_SYSTEM_PROMPT_DIR = "system"
_RUNTIME_PROMPT_DIR = "runtime"
_PROMPT_TEMPLATE_NAMES = [
    "generate_initial_prompt",
    "generate_user_response",
    "generate_test_status",
    "generate_evaluation",
]

_OPTIONAL_PROMPT_TEMPLATE_NAMES = [
    "generate_fail_follow_up",
]

# enable backwards-compatible StrEnum
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class TestStatusCategories(StrEnum):
    ALL_STEPS_ATTEMPTED = "A"
    NOT_ALL_STEPS_ATTEMPTED = "B"


class EvaluationCategories(StrEnum):
    ALL_EXPECTED_RESULTS_OBSERVED = "A"
    NOT_ALL_EXPECTED_RESULTS_OBSERVED = "B"


class Results(StrEnum):
    MAX_TURNS_REACHED = "Maximum turns reached."
    ALL_EXPECTED_RESULTS_OBSERVED = (
        "All of the expected results can be observed in the conversation."
    )
    NOT_ALL_EXPECTED_RESULTS_OBSERVED = (
        "Not all of the expected results can be observed in the conversation."
    )


class CanonicalEvaluator(BaseEvaluator):
    """An evaluator based on the canoncial templates. Compatible with the model providers supported in BedrockModelConfig"""

    def __init__(
        self,
        **kwargs,
    ):
        """Initialize the evaluator."""
        super().__init__(**kwargs)

        if self.template_root:
            # Create a custom Jinja template environment that uses the FileSystemLoader rather than the PackageLoader
            template_env = Environment(
                loader=FileSystemLoader(self.template_root),
                autoescape=jinja_env.autoescape,
            )
            template_prefix = ""

            # Check if generate_initial_prompt exists; if not, enable passthrough mode
            # where steps are sent directly to the agent without LLM processing.
            try:
                template_env.get_template(
                    f"{_RUNTIME_PROMPT_DIR}/generate_initial_prompt.jinja"
                )
                self._passthrough_steps = False
            except TemplateNotFound:
                self._passthrough_steps = True
                logger.debug(
                    "generate_initial_prompt template not found in custom template "
                    "directory. Steps will be passed directly to the agent "
                    "(passthrough mode)."
                )
        else:
            template_env = jinja_env
            template_prefix = f"{_PROMPT_TEMPLATE_ROOT}/"
            self._passthrough_steps = False

        # In passthrough mode, only generate_evaluation is required since steps
        # are sent directly to the agent and test_status is tracked by index.
        required_templates = (
            ["generate_evaluation"]
            if self._passthrough_steps
            else _PROMPT_TEMPLATE_NAMES
        )

        # Load required templates
        self._prompt_template_map = {
            name: {
                "system": template_env.get_template(
                    f"{template_prefix}{_SYSTEM_PROMPT_DIR}/{name}.jinja"
                ),
                "prompt": template_env.get_template(
                    f"{template_prefix}{_RUNTIME_PROMPT_DIR}/{name}.jinja"
                ),
            }
            for name in required_templates
        }

        # Load optional templates (if they exist)
        for name in _OPTIONAL_PROMPT_TEMPLATE_NAMES:
            try:
                # Try to load runtime template (required for optional templates)
                runtime_template = template_env.get_template(
                    f"{template_prefix}{_RUNTIME_PROMPT_DIR}/{name}.jinja"
                )

                # Try to load system template (optional for some templates)
                system_template = None
                try:
                    system_template = template_env.get_template(
                        f"{template_prefix}{_SYSTEM_PROMPT_DIR}/{name}.jinja"
                    )
                except Exception:
                    logger.debug(f"No system template for {name} (not required)")

                self._prompt_template_map[name] = {
                    "system": system_template,
                    "prompt": runtime_template,
                }
                logger.debug(f"Loaded optional template: {name}")
            except Exception as e:
                logger.debug(f"Optional template {name} not found: {e}")
                # Optional template not available, skip without error

    @staticmethod
    def _extract_content_from_xml(xml_data: str, element_names: list[str]) -> Tuple:
        content = []
        for e in element_names:
            pattern = rf"<{e}>(.*?)</{e}>"
            match = re.search(pattern, xml_data, re.DOTALL)
            content.append(match.group(1).strip() if match else "")
        return tuple(content)

    def _generate(
        self,
        system_prompt: str,
        prompt: str,
        output_xml_element: str,
    ) -> str:
        request_body = BedrockRequestHandler.build_request_body(
            request_body=self.model_config.request_body,
            model_config=self.model_config,
            system_prompt=system_prompt,
            prompt=prompt,
        )

        response = self.invoke_model(request_body=request_body)

        completion = BedrockRequestHandler.parse_completion_from_response(
            response=response, model_config=self.model_config
        )

        logger.debug(
            f"[{self.test.name}]\n[PROMPT]\n{prompt}\n[COMPLETION]\n{completion}"
        )

        output, reasoning = self._extract_content_from_xml(
            completion, [output_xml_element, "thinking"]
        )

        return output, reasoning

    def _generate_initial_prompt(self) -> str:
        system_prompt = self._prompt_template_map["generate_initial_prompt"][
            "system"
        ].render()
        prompt = self._prompt_template_map["generate_initial_prompt"]["prompt"].render(
            step=self.test.steps[0]
        )

        initial_prompt, reasoning = self._generate(
            system_prompt=system_prompt,
            prompt=prompt,
            output_xml_element="initial_prompt",
        )

        self.trace.add_step(
            system_prompt=system_prompt,
            prompt=prompt,
            initial_prompt=initial_prompt,
            reasoning=reasoning,
        )
        return initial_prompt

    def _generate_test_status(self) -> str:
        system_prompt = self._prompt_template_map["generate_test_status"][
            "system"
        ].render()
        prompt = self._prompt_template_map["generate_test_status"]["prompt"].render(
            steps=self.test.steps, conversation=self.conversation
        )
        test_status, reasoning = self._generate(
            system_prompt=system_prompt,
            prompt=prompt,
            output_xml_element="category",
        )
        self.trace.add_step(
            system_prompt=system_prompt,
            prompt=prompt,
            test_status=test_status,
            reasoning=reasoning,
        )
        return test_status

    def _generate_evaluation(self) -> tuple[str, str]:
        system_prompt = self._prompt_template_map["generate_evaluation"][
            "system"
        ].render()
        prompt = self._prompt_template_map["generate_evaluation"]["prompt"].render(
            expected_results=self.test.expected_results,
            conversation=self.conversation,
        )

        evaluation, reasoning = self._generate(
            system_prompt=system_prompt,
            prompt=prompt,
            output_xml_element="category",
        )
        self.trace.add_step(
            system_prompt=system_prompt,
            prompt=prompt,
            evaluation=evaluation,
            reasoning=reasoning,
        )

        return evaluation, reasoning

    def _generate_fail_follow_up(self, failure_reasoning: str) -> str:
        """Generate a follow-up question for the agent after a test failure.

        Args:
            failure_reasoning: The reasoning why the test failed

        Returns:
            The follow-up question to ask the agent
        """
        if "generate_fail_follow_up" not in self._prompt_template_map:
            logger.debug("No follow-up template available")
            return None

        # Just render the runtime template directly - no LLM generation needed
        follow_up_question = self._prompt_template_map["generate_fail_follow_up"]["prompt"].render(
            failure_reasoning=failure_reasoning,
            expected_results=self.test.expected_results,
            conversation=self.conversation,
        )

        self.trace.add_step(
            follow_up_question=follow_up_question,
            note="Direct template rendering - no LLM generation",
        )
        return follow_up_question

    def _generate_user_response(self) -> str:
        system_prompt = self._prompt_template_map["generate_user_response"][
            "system"
        ].render()
        prompt = self._prompt_template_map["generate_user_response"]["prompt"].render(
            steps=self.test.steps, conversation=self.conversation
        )

        user_response, reasoning = self._generate(
            system_prompt=system_prompt,
            prompt=prompt,
            output_xml_element="user_response",
        )

        self.trace.add_step(
            system_prompt=system_prompt,
            prompt=prompt,
            user_response=user_response,
            reasoning=reasoning,
        )
        return user_response

    def _invoke_target(self, user_input) -> str:
        target_response = self.target.invoke(user_input)
        self.trace.add_step(data=target_response.data)

        return target_response.response

    def _evaluate_and_handle_result(self) -> tuple:
        """Run evaluation and handle follow-up for failures.

        Returns:
            Tuple of (passed, result, reasoning, follow_up_response)
        """
        eval_category, reasoning = self._generate_evaluation()
        follow_up_response = None

        if (
            eval_category
            == EvaluationCategories.NOT_ALL_EXPECTED_RESULTS_OBSERVED.value  # noqa: W503
        ):
            result = Results.NOT_ALL_EXPECTED_RESULTS_OBSERVED.value

            # Try to generate follow-up question if template is available
            follow_up_question = self._generate_fail_follow_up(reasoning)

            if follow_up_question:
                try:
                    follow_up_response = self._invoke_target(follow_up_question)
                    self.conversation.add_turn(follow_up_question, follow_up_response)
                except Exception as e:
                    logger.warning(f"Failed to get follow-up response: {e}")
                    follow_up_response = None

            return False, result, reasoning, follow_up_response
        else:
            result = Results.ALL_EXPECTED_RESULTS_OBSERVED.value
            return True, result, reasoning, None

    def _evaluate_passthrough(self) -> TestResult:
        """Evaluate by sending each step directly to the agent.

        In passthrough mode, steps are used verbatim as agent prompts without
        LLM processing, and test status is tracked by step index rather than
        LLM classification. Only generate_evaluation is called (to judge results).

        Returns:
            TestResult
        """
        passed = False
        result = Results.MAX_TURNS_REACHED.value
        reasoning = ""
        follow_up_response = None
        all_steps_sent = False

        for step_idx, step in enumerate(self.test.steps):
            if self.conversation.turns >= self.test.max_turns:
                break

            # For the first step, honor initial_prompt override if provided
            if step_idx == 0 and self.test.initial_prompt:
                user_input = self.test.initial_prompt
            else:
                user_input = step

            self.conversation.add_turn(user_input, self._invoke_target(user_input))
        else:
            # Loop completed without break — all steps were sent
            all_steps_sent = True

        if all_steps_sent:
            passed, result, reasoning, follow_up_response = (
                self._evaluate_and_handle_result()
            )

        return TestResult(
            test_name=self.test.name,
            passed=passed,
            result=result,
            reasoning=reasoning,
            conversation=self.conversation,
            follow_up_response=follow_up_response,
        )

    def _evaluate_standard(self) -> TestResult:
        """Evaluate using LLM-driven step processing (original behavior).

        Returns:
            TestResult
        """
        passed = False
        result = Results.MAX_TURNS_REACHED.value
        reasoning = ""
        follow_up_response = None

        while self.conversation.turns < self.test.max_turns:
            if self.conversation.turns == 0:
                # start conversation
                if self.test.initial_prompt:
                    user_input = self.test.initial_prompt
                else:
                    user_input = self._generate_initial_prompt()
            else:
                # generate next user response
                user_input = self._generate_user_response()

            # add turn to the conversation
            self.conversation.add_turn(user_input, self._invoke_target(user_input))

            # get test status
            test_status = self._generate_test_status()
            if test_status == TestStatusCategories.ALL_STEPS_ATTEMPTED:
                # evaluate conversation
                passed, result, reasoning, follow_up_response = (
                    self._evaluate_and_handle_result()
                )
                break

        return TestResult(
            test_name=self.test.name,
            passed=passed,
            result=result,
            reasoning=reasoning,
            conversation=self.conversation,
            follow_up_response=follow_up_response,
        )

    def evaluate(self) -> TestResult:
        """Conduct the test.

        In passthrough mode (when generate_initial_prompt template is missing
        from the custom template directory), steps are sent directly to the
        agent without LLM processing, saving evaluator LLM costs.

        Returns:
            TestResult
        """
        if self._passthrough_steps:
            return self._evaluate_passthrough()
        else:
            return self._evaluate_standard()

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from agenteval.evaluators.canonical.evaluator import CanonicalEvaluator
from agenteval.evaluators.model_config.preconfigured_model_configs import (
    DEFAULT_CLAUDE_3_MODEL_CONFIG,
)
from agenteval.test import Test
from src.agenteval.utils import aws


class TestCustomTemplates:
    """Integration tests for custom template functionality."""

    @pytest.fixture
    def test_fixture(self):
        return Test(
            name="custom_template_test",
            steps=["step 1"],
            expected_results=["result 1"],
            initial_prompt="test prompt",
            max_turns=2,
        )

    @pytest.fixture
    def target_fixture(self):
        return MagicMock()

    @pytest.fixture
    def custom_templates_dir(self):
        """Create temporary custom templates for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create custom template directory structure
            custom_root = os.path.join(temp_dir, "custom_templates")
            system_dir = os.path.join(custom_root, "system")
            runtime_dir = os.path.join(custom_root, "runtime")

            os.makedirs(system_dir, exist_ok=True)
            os.makedirs(runtime_dir, exist_ok=True)

            # Create minimal custom templates (simplified versions for testing)
            templates = {
                "generate_initial_prompt.jinja": "Custom initial prompt template",
                "generate_user_response.jinja": "Custom user response template",
                "generate_test_status.jinja": "Custom test status template",
                "generate_evaluation.jinja": "Custom evaluation template",
            }

            # Write templates to both system and runtime directories
            for template_name, content in templates.items():
                for directory in [system_dir, runtime_dir]:
                    template_path = os.path.join(directory, template_name)
                    with open(template_path, 'w') as f:
                        f.write(content)

            yield custom_root

    def test_custom_templates_are_loaded(self, mocker, test_fixture, target_fixture, custom_templates_dir):
        """Test that custom templates are actually loaded and used."""
        # Mock AWS components
        mock_session = mocker.patch.object(aws.boto3, "Session")
        mocker.patch.object(mock_session.return_value, "client")

        # Create evaluator with custom template root
        evaluator = CanonicalEvaluator(
            aws_profile="test-profile",
            aws_region="us-west-2",
            endpoint_url=None,
            model_config=DEFAULT_CLAUDE_3_MODEL_CONFIG,
            test=test_fixture,
            target=target_fixture,
            work_dir="test_dir",
            template_root=custom_templates_dir,
        )

        # Verify that the template objects were created
        assert evaluator._prompt_template_map is not None
        assert len(evaluator._prompt_template_map) == 4

        # Verify all expected template names are present
        expected_templates = [
            "generate_initial_prompt",
            "generate_user_response",
            "generate_test_status",
            "generate_evaluation",
        ]
        for template_name in expected_templates:
            assert template_name in evaluator._prompt_template_map
            assert "system" in evaluator._prompt_template_map[template_name]
            assert "prompt" in evaluator._prompt_template_map[template_name]

    def test_template_root_fallback_to_default(self, mocker, test_fixture, target_fixture):
        """Test that None template_root falls back to default canonical templates."""
        # Mock AWS components
        mock_session = mocker.patch.object(aws.boto3, "Session")
        mocker.patch.object(mock_session.return_value, "client")

        # Create evaluator without custom template root
        evaluator = CanonicalEvaluator(
            aws_profile="test-profile",
            aws_region="us-west-2",
            endpoint_url=None,
            model_config=DEFAULT_CLAUDE_3_MODEL_CONFIG,
            test=test_fixture,
            target=target_fixture,
            work_dir="test_dir",
            template_root=None,  # Should use default
        )

        # Verify that the template objects were created (using default templates)
        assert evaluator._prompt_template_map is not None
        assert len(evaluator._prompt_template_map) == 4
        assert evaluator._passthrough_steps is False

    @pytest.fixture
    def passthrough_templates_dir(self):
        """Create a custom template directory with only generate_evaluation.

        When generate_initial_prompt is missing, the evaluator enters
        passthrough mode and sends steps directly to the agent.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_root = os.path.join(temp_dir, "passthrough_templates")
            system_dir = os.path.join(custom_root, "system")
            runtime_dir = os.path.join(custom_root, "runtime")

            os.makedirs(system_dir, exist_ok=True)
            os.makedirs(runtime_dir, exist_ok=True)

            # Only create generate_evaluation templates (no generate_initial_prompt,
            # generate_user_response, or generate_test_status)
            for directory in [system_dir, runtime_dir]:
                with open(os.path.join(directory, "generate_evaluation.jinja"), 'w') as f:
                    f.write("Evaluation template")

            yield custom_root

    def test_passthrough_mode_when_initial_prompt_template_missing(
        self, mocker, test_fixture, target_fixture, passthrough_templates_dir
    ):
        """Test that missing generate_initial_prompt triggers passthrough mode."""
        mock_session = mocker.patch.object(aws.boto3, "Session")
        mocker.patch.object(mock_session.return_value, "client")

        evaluator = CanonicalEvaluator(
            aws_profile="test-profile",
            aws_region="us-west-2",
            endpoint_url=None,
            model_config=DEFAULT_CLAUDE_3_MODEL_CONFIG,
            test=test_fixture,
            target=target_fixture,
            work_dir="test_dir",
            template_root=passthrough_templates_dir,
        )

        assert evaluator._passthrough_steps is True
        # Only generate_evaluation should be loaded
        assert "generate_evaluation" in evaluator._prompt_template_map
        assert "generate_initial_prompt" not in evaluator._prompt_template_map
        assert "generate_user_response" not in evaluator._prompt_template_map
        assert "generate_test_status" not in evaluator._prompt_template_map

    def test_passthrough_mode_not_triggered_with_all_templates(
        self, mocker, test_fixture, target_fixture, custom_templates_dir
    ):
        """Test that passthrough mode is NOT activated when all templates exist."""
        mock_session = mocker.patch.object(aws.boto3, "Session")
        mocker.patch.object(mock_session.return_value, "client")

        evaluator = CanonicalEvaluator(
            aws_profile="test-profile",
            aws_region="us-west-2",
            endpoint_url=None,
            model_config=DEFAULT_CLAUDE_3_MODEL_CONFIG,
            test=test_fixture,
            target=target_fixture,
            work_dir="test_dir",
            template_root=custom_templates_dir,
        )

        assert evaluator._passthrough_steps is False
        assert len(evaluator._prompt_template_map) == 4
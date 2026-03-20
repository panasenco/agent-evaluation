![PyPI - Version](https://img.shields.io/pypi/v/agent-evaluation)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/agent-evaluation)
![GitHub License](https://img.shields.io/github/license/awslabs/agent-evaluation)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Built with Material for MkDocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)

# Agent Evaluation

Agent Evaluation is a generative AI-powered framework for testing virtual agents.

Internally, Agent Evaluation implements an LLM agent (evaluator) that will orchestrate conversations with your own agent (target) and evaluate the responses during the conversation.

## ✨ Key features

- Built-in support for popular AWS services including [Amazon Bedrock](https://aws.amazon.com/bedrock/), [Amazon Q Business](https://aws.amazon.com/q/business/), and [Amazon SageMaker](https://aws.amazon.com/sagemaker/). You can also [bring your own agent](https://awslabs.github.io/agent-evaluation/targets/custom_targets/) to test using Agent Evaluation.
- Orchestrate concurrent, multi-turn conversations with your agent while evaluating its responses.
- Define [hooks](https://awslabs.github.io/agent-evaluation/hooks/) to perform additional tasks such as integration testing.
- Can be incorporated into CI/CD pipelines to expedite the time to delivery while maintaining the stability of agents in production environments.

## 📚 Documentation

To get started, please visit the full documentation [here](https://awslabs.github.io/agent-evaluation/). To contribute, please refer to [CONTRIBUTING.md](./CONTRIBUTING.md)

## 🚀 Passthrough Mode (Cost Optimization)

When your test steps are already phrased as exact questions for the agent (e.g. `"What if I need my 1099 sent to a different address?"`), you can skip the evaluator LLM calls that would otherwise rephrase your steps. This **passthrough mode** sends steps directly to the target agent, saving 3 LLM calls per conversation turn (`generate_initial_prompt`, `generate_user_response`, and `generate_test_status`). Only `generate_evaluation` is still called to judge the results.

### How to enable

If you are using a custom `template_root` in your evaluator config, simply **remove** (or don't create) the `generate_initial_prompt.jinja` file from both the `system/` and `runtime/` subdirectories. Keep `generate_evaluation.jinja` in both directories since it's still needed to judge pass/fail.

Your custom template directory should look like:

```
my_templates/
├── system/
│   └── generate_evaluation.jinja
└── runtime/
    └── generate_evaluation.jinja
```

And your test plan YAML:

```yaml
evaluator:
  model: claude-3_7-us
  template_root: /path/to/my_templates

tests:
  test_1099_address:
    steps:
      - "What if I need my 1099 sent to a different address?"
    expected_results:
      - "Agent explains how to update the 1099 mailing address"
    max_turns: 3
```

### Behavior

| Mode | Trigger | Steps handling | LLM calls per turn |
|---|---|---|---|
| **Standard** | All 4 templates present (or no `template_root`) | LLM rephrases steps into agent prompts | 3–4 (initial/response + status + evaluation) |
| **Passthrough** | `generate_initial_prompt.jinja` missing from custom `template_root` | Steps sent verbatim to agent | 1 (evaluation only) |

In passthrough mode:
- Each step is sent to the agent in order, one per turn.
- If `initial_prompt` is set on a test, it overrides the first step (same as standard mode).
- If `max_turns` is reached before all steps are sent, the test fails with "Maximum turns reached."
- The `generate_evaluation` template still runs to judge whether expected results were observed.

## 👏 Contributors

Shout out to these awesome contributors:

<a href="https://github.com/awslabs/agent-evaluation/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=awslabs/agent-evaluation" />
</a>
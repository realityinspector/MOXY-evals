# MOXY - Single-Command LLM Evaluation

MOXY (Mirascope + Oxen + YAML) is a lightweight, single-command evaluation tool for LLMs that combines:

- **Mirascope** for LLM API access
- **Oxen.ai** for version control of evaluation data
- **YAML** for rich, annotated output with anchors and aliases

## Quick Start

```bash
# Install dependencies
pip install mirascope tiktoken pyyaml rich oxenai

# Set API keys
export OPENAI_API_KEY="your-key-here"
export OXEN_API_KEY="your-key-here"  # Optional

# Run evaluation
python main.py
```

## Features

- 🚀 **Zero Configuration**: Run evaluations with just `python main.py`
- 📝 **Rich Output Formats**: JSON, Markdown, and YAML with annotations
- 🔄 **Version Control**: Built-in integration with Oxen.ai
- ⚙️ **Configurable Prompts**: Simple YAML config for custom evaluations
- 📊 **Automatic Metrics**: Token counts and timing built-in

## The Power of YAML Annotations

MOXY leverages YAML's unique features that JSON and other formats lack:

### 1. Anchors (`&`) and Aliases (`*`)

MOXY uses YAML anchors and aliases to create a true directed graph of your evaluation data:

```yaml
# Common configuration defined once
common: &common_config
  model: gpt-4o-mini
  timestamp: 2025-05-03T12:34:56
  token_usage: 300

# Each result references the common configuration
results:
  - <<: *common_config  # Merge in all common values
    input: {"country": "France"}
    output: |
      Paris is the capital of France.
```

This eliminates duplication and creates a true graph structure in memory - unlike JSON which requires full redundancy.

### 2. Human-Readable Comments

YAML allows comments anywhere in the data, preserved for humans but ignored by parsers:

```yaml
# This entire evaluation was run on May 3rd, 2025
eval_id: capital_test

results:
  - <<: *common_config
    # Result improved 15% over previous version
    input: {"country": "France"}
```

This lets you annotate your evaluation data with observations, explanations, and metadata without changing the underlying structure.

### 3. Multi-line String Preservation

LLM outputs often contain line breaks and formatting. YAML's pipe syntax (`|`) preserves exact formatting:

```yaml
output: |
  This paragraph has
  preserved line breaks
  exactly as the LLM
  generated them.
```

## Configuration

MOXY uses a simple `config.yaml` file to define evaluations:

```yaml
capital_test:
  prompt: "What is the capital of {country}?"
  inputs: 
    - {"country": "France"}
    - {"country": "Japan"}
    - {"country": "Brazil"}

# Add more evaluation configurations as needed
summarization_test:
  prompt: "Summarize the following text: {text}"
  inputs:
    - {"text": "Long text goes here..."}
```

## Output Files

For each evaluation run, MOXY generates:

- `results/[eval_id]/[timestamp]_[uuid].json` - Traditional JSON output
- `results/[eval_id]/[timestamp]_[uuid].md` - Human-readable Markdown
- `results/[eval_id]/[timestamp]_[uuid].yaml` - Rich YAML with anchors and comments

## Oxen.ai Integration

MOXY automatically:

1. Initializes a local Oxen repository
2. Creates a branch for each evaluation type
3. Commits evaluation results with descriptive messages
4. Pushes to Oxen.ai if credentials are provided
5. Provides clickable links to view results online

To enable, set:

```bash
export OXEN_API_KEY="your-key-here"
export OXEN_REMOTE="https://hub.oxen.ai/username/repo"  # Optional
```

## Advanced Usage

### Custom Evaluations

Edit `config.yaml` to add new evaluation types:

```yaml
my_new_eval:
  prompt: "Your custom prompt with {placeholders}"
  inputs:
    - {"placeholders": "values"}
```

### Extending YAML Output

The `_mk_yaml()` function can be modified to add custom anchors, comments, or structural elements to your YAML output.

## Why YAML for Evaluation Data?

1. **Documentation via Comments**: Add rationale, observations, and metadata
2. **Reduced Redundancy**: Use anchors/aliases to eliminate duplicate information
3. **Readability**: More human-readable than JSON, especially for complex data
4. **Format Preservation**: Better handling of multiline text from LLM outputs
5. **Extensibility**: Easy to add new metadata without breaking parsers

## Contributing

Contributions welcome! Feel free to open issues or PRs for new features or improvements.
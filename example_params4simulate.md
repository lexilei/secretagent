# Example File for Simulate

One JSON file per ptools. Top-level keys = interface names; each value = list of `{input_args, expected_output}`.

## JSON Format

```json
{
  "sport_for": [
    {"input_args": ["Bam Adebayo"], "expected_output": "basketball"},
    {"input_args": ["scored a reverse layup"], "expected_output": "basketball"}
  ],
  "analyze_sentence": [
    {"input_args": ["Bam Adebayo scored a reverse layup."], "expected_output": ["Bam Adebayo", "scored a reverse layup", ""]}
  ]
}
```

## Config

Put the file in the benchmark dir (e.g. `benchmarks/sports_understanding/examples/examples.json`). Path is relative to benchmark root.

```yaml
ptools:
  sport_for:
    method: simulate
    example_file: examples/examples.json
  analyze_sentence:
    method: simulate
    example_file: examples/examples.json
```

Each interface reads its examples via `data[interface.name]`. Missing key = no examples.

# Project maintenance instructions

## Version every completed update

- The user designated version 1.0 as "Minimum core functionality met".
- `VERSION` is the source of truth. For every future completed edit or improvement, increment the second integer by one: 1.0, 1.1, 1.2, and 1.10 after 1.9. Bump once per completed logical update, not per file or keystroke.
- Update `VERSION`, the README version line and its release-history table together. Summarize the actual change and run checks appropriate to it.
- When publishing an authorized update, create an annotated `v<version>` tag on its final commit and push that tag with the branch. Never move or overwrite an existing tag. If a tag already exists, inspect it before choosing the next version.
- Keep confidential data, identity maps, model artifacts, scores, outputs and the Python environment out of Git. Preserve the private identity map and partition registry locally.
- Keep user manuals and assessment reports outside this coding folder. Maintain the README here as the repository entry point.
- Distinguish core software completion from scientific validity. Do not claim reliable incident prediction or assessment completion from passing software tests alone.

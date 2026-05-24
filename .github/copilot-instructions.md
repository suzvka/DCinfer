# Copilot Instructions

## General Guidelines
- In VS environment, when using Python in terminal, provide complete scripts in a single command; line-by-line input may hang with no output and time out.
- By default the user authorizes Copilot to search and read any files within the repository workspace when investigating or modifying code.

## C++ Engineering Guidelines
- Target users are traditional C++ engineers, but consider incorporating NumPy-style chained View-based tensor editing for enhanced usability.
- Utilize View index chaining to support NumPy-style operations, making tensor manipulation more intuitive for users familiar with this approach.
- Unify DC::Tensor data-block APIs around std::span; provide compatibility via overloads that accept std::vector (copy) or rvalue vector (move) into internal storage, while maintaining external access as span-based for performance and simplicity.

## DCinfer Architecture
- For the DCinfer refactor, ignore existing `DC::Infer` initially; create a new `InferNode` in new files first, and only later adapt `DC::Infer` into the new system.
- Design `InferNode` as an atomic input->compute->output node.
- Use `Sub` solely as a linear local orchestration unit with a local `TensorMap`.
- Ensure top-level schedules only manage `Sub` units.
- Make nodes stateful, reusable, and thread-safe, supporting multi-threading via task queues.
- Implement error handling that short-circuits into result objects instead of using exceptions.
- Clarify runtime semantics: nodes execute synchronously; concurrency control moves from node queues to `Sub` queues; shared node instances are allowed but must be invoked through blocking serialized access.
- Define `Sub` input/output schemas explicitly; top-level returns only the last `Sub` outputs, and failures short-circuit.
- Default tensors to move semantics, and ensure node/sub configurations are strong C++ types.
- Shared `InferNode` instances should carry their own execution lock.
- `Sub` serialization should be FIFO fair.
- Local tensor keys should avoid collisions by namespacing with node names.
- Top-level `Sub` output maps only require unique names.
- Explicit sub-to-sub mapping is preferred despite configuration complexity concerns.
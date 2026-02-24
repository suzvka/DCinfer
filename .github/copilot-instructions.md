# Copilot Instructions

## General Guidelines
- In VS environment, when using Python in terminal, provide complete scripts in a single command; line-by-line input may hang with no output and time out.
 - By default the user authorizes Copilot to search and read any files within the repository workspace when investigating or modifying code.

## C++ Engineering Guidelines
- Target users are traditional C++ engineers, but consider incorporating NumPy-style chained View-based tensor editing for enhanced usability.
- Utilize View index chaining to support NumPy-style operations, making tensor manipulation more intuitive for users familiar with this approach.
- Unify DC::Tensor data-block APIs around std::span; provide compatibility via overloads that accept std::vector (copy) or rvalue vector (move) into internal storage, while maintaining external access as span-based for performance and simplicity.
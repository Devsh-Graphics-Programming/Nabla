# Contributing to Nabla Engine

Thank you for your interest in contributing to the Nabla Engine! Nabla is a high-performance rendering framework that leverages Vulkan, OptiX, and CUDA. This document provides guidelines to help you contribute effectively.

## Table of Contents

- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Implementing Features](#implementing-features)
  - [Free Tasks to Take](#free-tasks-to-take)
- [Pull Request Process](#pull-request-process)
- [Connect with Other Project Contributors](#connect-with-other-project-contributors)
- [License](#license)

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please report it by opening an issue [here](https://github.com/Devsh-Graphics-Programming/Nabla/issues). Include details such as:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected and actual results
- Any relevant logs or screenshots

### Suggesting Enhancements

If you have ideas for enhancements, please submit them as issues [here](https://github.com/Devsh-Graphics-Programming/Nabla/issues). Describe:

- The feature you would like to see
- Why it would be beneficial
- Any relevant examples or mockups

### Implementing Features

We welcome pull requests for new features! See the [Free Tasks to Take](#free-tasks-to-take) section for a list of ideas. When submitting a pull request:

- Ensure your code compiles
- Ensure it runs without issues at runtime
- Update documentation as necessary

### Free Tasks to Take, DO ANY - WE CAN HIRE YOU :)

Below are some tasks that are open for contribution. If you start working on a task, please open an issue to let us know. *We can hire contributors if a pull request for the task is opened and merged by us*. Follow [Pull Request Process](#pull-request-process) if you decide to contribute. The list may get updated, new tasks may be added and already completed deleted, so before you start working on any please make sure it's still free.

- Extend `nbl::system` to download textures and meshes over HTTP/URL
- Mesh Shader Pipeline implementation and demo
- Raytracing Pipeline implementation and demo
- Perfect Hashing implementation
- Compute Frustum Culling in HLSL with Nabla Chad Append
- Merge Sort (Vulkan Compute)
- Radix Sort (Vulkan Compute)
- Bitonic Sort (Vulkan Compute)
- Merge Path (Vulkan Compute)
- [GPU flip fluid simulation task idea](https://www.youtube.com/watch?v=okQzAJM7LcE) -> discord [thread](https://discord.com/channels/318590007881236480/374061825454768129/1257988553112027186) for more info  

 **[Issues](https://github.com/Devsh-Graphics-Programming/Nabla/issues) also count as free tasks.** We also have old issues from previous generation **[here](https://github.com/buildaworldnet/IrrlichtBAW/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)**.

If you have questions about the list, you can ask directly on our [Discord server](https://discord.gg/SsJzqS23) or open an issue [here](https://github.com/Devsh-Graphics-Programming/Nabla/issues).

## Pull Request Process

1. **Do not** open pull requests which "fix" code formatting, the use of `NULL`, C-style casts, C++11, exceptions, or directory structure. This is a "DO NOT WANT"(TM) ESPECIALLY for your first pull requests. If your need to do this is so burning, then provide us with a pull request containing a script to do this instead.
2. Ensure you follow our [Implementing Features](#implementing-features) standards.
3. Split your larger contributions into as many separate independent features as practical.
4. Fork from the `master` branch or the last stable tag (with exceptions for some features).
5. Develop your contribution in your fork or a separate branch.
6. Comment your code, even just a tiny bit.
7. Supply an example which also serves as a test to the `examples_test` folder so that we can see how to use your contribution and that it works.
8. Open a pull request with just the code relevant to that one feature.
9. Engage in discussion in the pull request and polish your submission while taking some really minor and mild constructive criticism.
10. Get your pull request merged.
11. If in doubt, look at point 1.

## Connect with Other Project Contributors

Join our [Discord server](https://discord.gg/SsJzqS23) to connect with other contributors. It's the first place to go for questions and discussions about the project.

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).

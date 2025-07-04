# MoVer: Motion Verification for Motion Graphics Animations

[Jiaju Ma](https://majiaju.io) and
[Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/)
<br />
In ACM Transactions on Graphics (SIGGRAPH), 44(4), August 2025. To Appear.
<br />

![MoVer Teaser](assets/mover_teaser.png)
<br />

[![arXiv](https://img.shields.io/badge/arXiv-2502.13372-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2502.13372)
[![PyPI - Version](https://img.shields.io/pypi/v/mover?style=flat-square)](https://pypi.org/project/mover/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square)](LICENSE)


This repository contains the official implementation of MoVer, a domain-specific language based on first-order logic that can verify if various spatio-temporal properties of motion graphics are satisfied by an animation. We provide tools to use MoVer as part of a motion graphics animation generation pipeline with verification.

Check out the [project page](https://mover-dsl.github.io/) for animation and benchmark results.


## Dataset
The MoVer dataset of 5,600 prompts used in the paper can be found in `mover_dataset/`. Each prompt contains the ground truth MoVer program and information about the prompt's syntactic construction and whether an LLM is used as part of its generation.


## Installation
1. Set up a virtual environment with your favorite tool (`python>=3.10`). We recommend `uv` for its speed.
    ```bash
    uv venv mover_env --python 3.12 # or your favorite tool, like conda, venv, etc.
    ```

2. Make sure you have the right version of `pytorch` installed for your system (see [pytorch.org](https://pytorch.org/get-started/locally/)).

3. Install MoVer with `pip`
    ```bash
    pip install git+https://github.com/jama1017/Jacinle.git && \
    pip install git+https://github.com/jama1017/Concepts.git && \
    pip install mover
    ```

    or with `uv`
    ```bash
    uv pip install git+https://github.com/jama1017/Jacinle.git && \
    uv pip install git+https://github.com/jama1017/Concepts.git && \
    uv pip install mover
    ```

4. Alternatively, to install MoVer for development, first clone this repository
    ```bash
    git clone https://github.com/jama1017/MoVer.git
    cd MoVer
    ```

    and then install the editable version of MoVer
    ```bash
    pip install -r requirements.txt && \
    pip install -e .

    # or with uv
    uv pip install -r requirements.txt && \
    uv pip install -e .
    ```

5. To install the necessary browser for MoVer converter, run
    ```bash
    playwright install
    ```

6. Based on what models you plan to use, add your API keys as environment variables.
    - They must be named as `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `GROQ_API_KEY`.
    - MoVer includes APIs to interface with OpenAI, Gemini (via [OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)), and [Groq](https://groq.com/) by default.

7. (Optional) If you plan to run locally-hosted models, install the following dependencies:
    - Install `ollama` if you plan to use ollama.
    - Install `vLLM` if you plan to use vLLM.

8. (Optional) We support rendering animations to video with OpenCV as `.mp4` files, but the video produced might have limited compatibility because of codec issues. If `ffmpeg` is installed on your system, our converter will automatically use it to convert rendered videos.


## Quick Start
![MoVer Pipeline](assets/mover_pipeline.png)
<br />

### Starter Example
Once you have installed MoVer, clone this repository to get access to the `examples/` directory, where we have prepared some examples for you to try out. By default, OpenAI models are used, so make sure you have stored your API key as environment variables (must be named `OPENAI_API_KEY`). Or you can change the config file to use other models (see `examples/configs/` for examples).

First, to get things started, from the root directory of this repository, run the following command to generate some simple animations with the MoVer pipeline (using [`examples/configs/config_starter.yaml`](examples/configs/config_starter.yaml)): 
```bash
python -m mover.pipeline examples/configs/config_starter.yaml
```
The MoVer pipeline takes in a YAML config file as input. Here, the prompts used are stored in [`examples/prompts/prompts_starter.json`](examples/prompts/prompts_starter.json).
If you look at the JSON file, you can see that, for each prompt, we have populated the `ground_truth_program` field with the MoVer program for verification.
Running this command should create a directory called `example_output/prompts_starter/`, where you can find the iterations of generated animations with videos.


### Teaser Hi Example
<img src="assets/teaser_hi_animation.gif" width="400"/>

Next, let's recreate the teaser Hi example in the paper by running the following command (pre-generated results are available in `examples/`).
```bash
python -m mover.pipeline examples/configs/config_teaser.yaml
```
This time, we did not fill in the `ground_truth_program` field in the prompts, so the pipeline will generate a MoVer program for verification and store it in the `example_output/prompts_teaser/` directory as a Python script.

To create your own animations with MoVer, modify the starter config file and write your own prompts in the following format:
```json
[
    {
        "svg_name": "<name of the SVG file (then specify the directory in the config file)>",
        "svg_file_path": "<alternatively, you can specify the exact path to the SVG file>",
        "chat_id_name": "<unique identifier for the prompt>",
        "animation_prompt": "<describe the animation in detail. avoid fuzzy descriptions like 'make the square dance'>",
        "ground_truth_program": "(optional) <ground truth MoVer program>",
        "has_run": false
    }
]
```
Setting `has_run` to `true` will make the pipeline ignore this prompt.


## Usage Guide

### Tutorial
Check out the [tutorial.ipynb](tutorial.ipynb) for a walkthrough of each part of the MoVer pipeline (animation synthesis, MoVer program synthesis, and MoVer verification).

### SVG Animation
To understand how MoVer's animation synthesizer generates SVG animations using a simple JavaScript API based on [GSAP](https://gsap.com/), check out the synthesizer's system message [`sys_msg_animation_synthesizer.md`](src/mover/synthesizers/assets/sys_msg_animation_synthesizer.md) and the API itself in [`api.js`](src/mover/converter/assets/api.js).
- To extend the API, make sure to update [`api.js`](src/mover/converter/assets/api.js) and reflect the changes in the system message. See [`tutorial.ipynb`](tutorial.ipynb) for how to pass in your own system message.
- Each SVG animation is saved as an HTML file (see `examples/`). To properly render the HTML file, first get all the files in `src/mover/converter/assets/` and put them in the same directory as the HTML file. Then open the HTML file in your browser to see the animation in action.

### MoVer DSL
The MoVer DSL is designed with predicates corresponding to spatial-temporal concepts that people commonly use in natural language to describe motions. For example, for the following animation prompt:
> Translate the black square upwards by 100 px

We can write the corresponding MoVer program as:
```python
o_1 = iota(Object, lambda o: color(o, "black") and shape(o, "square"))
m_1 = iota(Motion, lambda m: type(m, "translate") and direction(m, [0.0, 1.0]) and magnitude(m, 100.0) and agent(m, o_1))
```

Table 1 in the [paper](https://arxiv.org/abs/2502.13372) gives an overview of the predicates in the MoVer DSL. For more detailed documentations and examples of how they can be composed into MoVer programs, check out MoVer synthesizer's system message [`sys_msg_mover_synthesizer.md`](src/mover/synthesizers/assets/sys_msg_mover_synthesizer.md), figures in the [paper](https://arxiv.org/abs/2502.13372), and the [results page](https://mover-dsl.github.io/#result-animations).
- To extend the DSL, update scripts in [`src/mover/dsl/`](src/mover/dsl/) and reflect the changes in the system message.


#### Resolving references to similar objects and motions
For the animation prompt below, notice that we have two black squares in the scene, as well as two rightward translation motions (see the SVG [here](examples/svg/two_black_squares.svg) and the generated animation [here](examples/two_black_squares_animation.mp4)).
>"Translate the first black square to the right, then down, and then to the right. Translate the second black square up."

To refer to the second instances of repeated objects and motions, we can use the `not` predicate to exclude the first instance. This pattern generalizes to more instances of repetitions as well. For example, for the above animation prompt, we can write the corresponding MoVer program as:
```python
o_1 = iota(Object, lambda o: color(o, "black") and shape(o, "square"))
## notice the use of not o_1 to refer to the second black square
o_2 = iota(Object, lambda o: color(o, "black") and shape(o, "square") and not o_1)

m_1 = iota(Motion, lambda m: type(m, "translate") and direction(m, [1.0, 0.0]) and agent(m, o_1))
m_2 = iota(Motion, lambda m: type(m, "translate") and direction(m, [0.0, -1.0]) and agent(m, o_1))
## notice the use of not m_1 to refer to the second rightward translation motion
m_3 = iota(Motion, lambda m: type(m, "translate") and direction(m, [1.0, 0.0]) and agent(m, o_1) and not m_1)
m_4 = iota(Motion, lambda m: type(m, "translate") and direction(m, [0.0, 1.0]) and agent(m, o_2))

t_before(m_1, m_2)
t_after(m_3, m_2)
```


## Release Checklist
- [x] MoVer DSL and verifier
- [x] SVG animation to MoVer data format converter
- [x] LLM-based animation synthesizer
- [x] LLM-based MoVer synthesizer
- [x] Scripts to run the full pipeline
- [ ] Chat-based UI and backend for creating animations with MoVer
- [ ] Scripts for generating animation prompts


## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Contact
Jiaju Ma<br />
[@jama1017](https://x.com/jama1017) <br />
[majiaju.io](https://majiaju.io)<br />
hellojiajuma@gmail.com<br />


## Acknowledgments
We thank [Yusong Wu](https://lukewys.github.io/) for his help on getting this repository ready for release. This project builds on the wonderful foundation of the [Concepts](https://github.com/concepts-ai/Concepts) framework by [Jiayuan Mao](https://jiayuanm.com/). The MoVer DSL parser and executor is based on [LEFT](https://github.com/joyhsu0504/LEFT) by [Joy Hsu](https://web.stanford.edu/~joycj/) and [Jiayuan Mao](https://jiayuanm.com/). Our SVG animation API uses the one and only [GSAP](https://gsap.com/). The MoVer converter uses the [ntc js](https://chir.ag/projects/ntc/) (Name that Color JavaScript) library for converting hex colors to names.
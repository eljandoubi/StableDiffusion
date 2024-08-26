<p align="center">
    <a href="docs/imgs/StableDiffusion-Model-Logo.jpg">
        <img src="docs/imgs/StableDiffusion-Model-Logo.jpg" width="50%"/>
    </a>
</p>

<p align="center">
    <a href="License"><img src="https://img.shields.io/github/license/eljandoubi/StableDiffusion"></a>
    <a href="Linux"><img src="https://img.shields.io/github/actions/workflow/status/eljandoubi/StableDiffusion/linux-conda.yml?label=Linux"></a>
    <a href="Conda"><img src="https://img.shields.io/github/actions/workflow/status/eljandoubi/StableDiffusion/linux-conda.yml?label=Conda"></a>
    <a href="Pylint"><img src="https://img.shields.io/github/actions/workflow/status/eljandoubi/StableDiffusion/pylint-test.yml?label=Pylint"></a>
    <a href="Pytest"><img src="https://img.shields.io/github/actions/workflow/status/eljandoubi/StableDiffusion/pytest-ci.yml?label=Pytest"></a>
</p>

Coding Stable Diffusion from scratch using pytorch.

## Setup environment
* Clone the repository and Go to StableDiffusion directory.
```bash
git clone https://github.com/eljandoubi/StableDiffusion.git && cd StableDiffusion
```

* Build environment.
```bash
make build
```
## Check the code sanity
```bash
make check
```
## Run the pipeline
```bash
make run
```
## Clean environment
```bash
make clean
```
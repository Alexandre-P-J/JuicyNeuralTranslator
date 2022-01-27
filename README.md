# Neural Web Translator
<img src="./Deploy/web/app/static/img/JT_logo_letra_horizontal.svg" alt="Web translation service logo" height="150"/><p>Translate text and .docx, .txt, .pdf documents using the state of the art in machine learning models for neural machine translation</p>

## Instructions
This software only requires [docker-compose](https://docs.docker.com/compose/install "Install docker-compose") and it should run on any platform supported by docker. Minimal modifications may be required to deploy the system with other orchestration solutions.
### Setup and execution:
 - This project runs entirely on docker containers, when using docker compose you can simply follow this instructions.
 - Download or clone this repository.
 - Open a terminal in the project directory or `cd THIS_PROJECT_FOLDER`
 - Run `docker-compose build`
 - It may take up to several minutes depending on your internet connection.
 - You may run the service with the following command: `docker-compose up`
### About this project
Tweets from a person of interest were collected and analyzed using docker. For this, various docker containers for different data processing tasks were created (see below). Those containers were set into a pipeline via a [docker-compose.yml]() file.

**Note: This project is not finished yet ...**

<br>


### The docker containers
Each container consists of at least a
- Dockerfile:
  - declaration of used python image
  - declaration of virtual working directory
  - command for pip installation
  - call of python file
- requirements.txt:
  - required python modules to be imported for processing the python file
- Python file with the relevant code


#### [tweet_collector]():
For accessing [Twitter](https://twitter.com/home) data you additionally need a [config.py](), which includes an access_token. You get this token, when you create a Twitter [developer account](https://developer.twitter.com/en). <br>





#### [mongo_db]():



#### [etl_job]():



#### [postgres_db]():

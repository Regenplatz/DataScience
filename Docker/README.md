### About this project
Tweets from a person of interest were collected and analyzed using docker. For this, various docker containers for different data processing tasks were created (see below). Those containers were set into a pipeline via a [docker-compose.yml]() file.
<br>

##### About the TwitterPipeline
This pipeline retreives tweets (see [tweet_collector](https://github.com/Regenplatz/DataScience/tree/master/Docker/TwitterPipeline/tweet_collector)), saves the data in a mongo database (see [mongo_db]()), transforms the data (see [etl_job]()) and finally forwards it to a postgres database (see [postgres_db]()).

<br>

### The docker containers
Each container consists of at least a
- `Dockerfile`
  - declaration of used python image
  - declaration of virtual working directory
  - command for pip installation
  - call of python file
- `requirements.txt`
  - required python modules to be imported for processing the python file
- `Python file` with the relevant source code


#### [tweet_collector](https://github.com/Regenplatz/DataScience/tree/master/Docker/TwitterPipeline/tweet_collector):
For accessing [Twitter](https://twitter.com/home) data you additionally need a [config.py](), which includes an access_token. You get this token, when you create a [Twitter developer account](https://developer.twitter.com/en). <br>



<br>

**Note: This project is not finished yet ...**<br>
**- Other containers need to be build.**<br>
**- docker-compose file needs to be created.**<br>
**- readme needs to be updated**



#### [mongo_db]():



#### [etl_job]():



#### [postgres_db]():

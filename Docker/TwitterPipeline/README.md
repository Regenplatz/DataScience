## About this project
Tweets from a person of interest were collected and analyzed using docker. For this, various docker containers for different data processing tasks were created (see below). Those containers were set into a pipeline via [docker-compose.yml](docker-compose.yml).
<br>

#### About the Twitter Pipeline
This pipeline retrieves tweets (see [tweet_collector](tweet_collector)) and saves the data to a mongo database. From there, tweets are transforms, analyzed for sentiment and finally forwards it to a postgres database (see [etl_job](etl_job)) .
<br>

#### About the Docker Containers
Each container consists of at least a
- `Dockerfile`
  - declaration of used python image
  - declaration of virtual working directory
  - command for pip installation
  - call of python file
- `requirements.txt`
  - required python modules to be imported for processing the python file
- `*.py` python file with the relevant source code ([tweety.py](tweet_collector/tweety.py) or [etl.py](etl_job/etl.py))

There is an additional file `config.py` in the container [tweet_collector](tweet_collector), which is needed to access tweets from twitter. It contains information about customer API and access token.

##### How to Use
- Ensure that [Docker](https://docs.docker.com/get-docker/) is installed.
- Get Twitter Developer API on [Developer.Twitter.com](https://developer.twitter.com/).
- Setup the config.py file with your API creds.
- Clone the repository - git clone https://github.com/Regenplatz/DataScience/tree/master/Docker/TwitterPipeline
- In your terminal, run *docker-compose up*.

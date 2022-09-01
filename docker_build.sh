docker_version=seals:v1.4
docker_gz=~/seals.tar.gz

sudo docker build -t $docker_version ./docker_init/

sudo docker save $docker_version | gzip -c > $docker_gz
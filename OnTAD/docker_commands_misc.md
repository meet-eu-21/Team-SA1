**GENERAL LINKS, NOTES**

https://docker-curriculum.com/ => good intro
https://www.digitalocean.com/community/tutorials/how-to-share-data-between-the-docker-container-and-the-host => mounting and binding volumes

# Delete all unused containers
docker rm $(docker ps -a -q -f status=exited)
# OR
docker container prune

# Note on the -v (mounting flag)
 Note: The -v flag is very flexible. It can bindmount or name a volume with just a slight adjustment in syntax. If the first argument begins with a / or ~/, you’re creating a bindmount. Remove that, and you’re naming the volume.

    -v /path:/path/in/container mounts the host directory, /path at the /path/in/container
    -v path:/path/in/container creates a volume named path with no relationship to the host.

**ONTAD**
https://github.com/anlin00007/OnTAD 

# Pulling image:
docker pull anlin00007/ontad:v1.2

# Most basic, run after image has been manually run in docker desktop (this doesn't do anything, though, just tells me
# that the matrix cannot be opened; naturally, since I haven't mapped it)
docker run --rm anlin00007/ontad:v1.2 OnTAD

# Attempt at running OnTAD container from image while mounting the OnTAD folder to the Docker container in the tmp folder
# This works! Now, need to get it working with calling OnTAD
docker run --rm -i -v /mnt/c/Users/yannz/OneDrive/Bureau/OnTAD:/tmp anlin00007/ontad:v1.2

# This works too, but I can't find the output file specified, so I'm going to run it again with an explicit BED file output
docker run -i -v /mnt/c/Users/yannz/OneDrive/Bureau/OnTAD:/tmp anlin00007/ontad:v1.2 OnTAD /tmp/data/chr18_KR.matrix -penalty 0.1 -maxsz 200 -o OnTAD_KRnorm_pen0.1_max200_chr18

# This works and puts the output file in my mapped folder
docker run -i -v /mnt/c/Users/yannz/OneDrive/Bureau/OnTAD:/tmp anlin00007/ontad:v1.2 OnTAD /tmp/data/chr18_KR.matrix -penalty 0.1 -maxsz 200 -o tmp/output/O
nTAD_KRnorm_pen0.1_max200_chr18

# This doesn't work for some reason => Error: chrnum is required and must be valid
docker run -i -v /mnt/c/Users/yannz/OneDrive/Bureau/OnTAD:/tmp anlin00007/ontad:v1.2 OnTAD /tmp/data/chr18_KR.matrix -penalty 0.1 -maxsz 200 -o tmp/output/O
nTAD_KRnorm_pen0.1_max200_chr18 -bedout chr18 78077248 10000

# Attempt to do on preprocessed Chr1 => this works too!
docker run -i -v /mnt/c/Users/yannz/OneDrive/Bureau/OnTAD:/tmp anlin00007/ontad:v1.2 OnTAD /tmp/data/chr1_100kb.matrix -penalty 0.1 -maxsz 200 -o tmp/output/OnTAD_KRnorm_pen0.1_max200_chr1_100kb

**TADBIT**
https://github.com/3DGenomes/TADbit/tree/v1.0.1 

# Pulling image:
docker pull vntasis/hicbit-tadbit:1.1
sudo docker run  -v /home/cwh/book:/book -v /home/cwh/blog:/blog node-gitbook gitbook install
sudo docker run  -v /home/cwh/book:/book -v /home/cwh/blog:/blog node-gitbook gitbook build /book /blog

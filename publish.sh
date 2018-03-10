docker run  -v /home/gitbook/book:/book -v /home/gitbook/blog:/blog node-gitbook gitbook install
docker run -v /home/gitbook/book:/book -v /home/gitbook/blog:/blog node-gitbook gitbook build /book /blog

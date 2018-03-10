#!/bin/sh
cd /home/gitbook/book
git pull
python3 gen_rss.py
docker run  -v /home/gitbook/book:/book -v /home/gitbook/blog:/blog node-gitbook gitbook install
docker run -v /home/gitbook/book:/book -v /home/gitbook/blog:/blog node-gitbook gitbook build /book /blog > build_blog.log

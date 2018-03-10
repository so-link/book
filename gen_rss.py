import re
import os
import time
import datetime
import PyRSS2Gen
_re_link = re.compile(r'\[(.*)\]\((.*)\)$')

BOOK_DIR = '/home/gitbook/book'
ROOT_URL = 'https://blog.so-link.org'

SUMMARY_MD_PATH = os.path.join(BOOK_DIR, 'SUMMARY.md')
POST_LIST_FILE_PATH = os.path.join(BOOK_DIR, 'posts/README.md')
RSS_XML_PATH = os.path.join(BOOK_DIR, 'rss.xml')
posts = []

with open(SUMMARY_MD_PATH) as file:
    for line in file:
        link = line.strip()
        match = _re_link.search(link)
        if match is None:
            continue
        title = match.group(1)
        path = match.group(2)
        if path.endswith('.md') and not path.endswith('README.md'):
            file_path = os.path.join(BOOK_DIR, path)
            mtime = time.localtime(os.stat(file_path).st_mtime)
            posts.append((title, path, mtime))

posts = list(set(posts))

posts.sort(key=lambda entry: (entry[2], entry[0]), reverse=True)

with open(POST_LIST_FILE_PATH, 'w') as file:
    for title, path, mtime in posts:
        file.write('- [{} {}](/{}.html)\n\n'.format(time.strftime('%Y-%m-%d', mtime), title, path[:-3]))

rss2items = []

for title, path, mtime in posts:
    rss2items.append(PyRSS2Gen.RSSItem(
        title = title,
        link = '{}/{}.html'.format(ROOT_URL, path[:-3]),
        guid = PyRSS2Gen.Guid(path),
        pubDate = datetime.datetime.fromtimestamp(time.mktime(mtime)),
    ))

rss = PyRSS2Gen.RSS2(
    title = 'blog.so-link.org',
    link = 'https://blog.so-link.org/',
    items = rss2items,
    description = 'solink blog',
    lastBuildDate = datetime.datetime.now(),
)

rss.write_xml(open(RSS_XML_PATH, 'w'))


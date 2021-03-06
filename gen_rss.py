import re
import os
import time
import datetime
import PyRSS2Gen
_re_link = re.compile(r'\[(.*)\]\((.*)\)$')

BOOK_DIR = '/home/gitbook/book'
BLOG_DIR = '/home/gitbook/blog'
ROOT_URL = 'https://blog.so-link.org'

SUMMARY_MD_PATH = os.path.join(BOOK_DIR, 'SUMMARY.md')
POST_LIST_FILE_PATH = os.path.join(BOOK_DIR, 'posts/README.md')
RSS_XML_PATH = os.path.join(BOOK_DIR, 'rss.xml')
IGNORE_PATH_PREFIX = [
    'posts/',
]

def ignored(path):
    for prefix in IGNORE_PATH_PREFIX:
        if path.startswith(prefix):
            return True
    return False

def md2url(link):
    if link.endswith('README.md'):
        return '/{}'.format(link[:-9])
    if link.endswith('.md'):
        return '/{}.html'.format(link[:-3])
    return link

posts = []

with open(SUMMARY_MD_PATH) as file:
    for line in file:
        try:
            link = line.strip()
            match = _re_link.search(link)
            if match is None:
                continue
            title = match.group(1)
            path = match.group(2)
            if path.endswith('.md') and not ignored(path):
                file_path = os.path.join(BOOK_DIR, path)
                mtime = time.localtime(os.stat(file_path).st_ctime)
                posts.append((title, path, mtime))
        except Exception as e:
            print(e)

posts = list(set(posts))

posts.sort(key=lambda entry: (entry[2], entry[0]), reverse=True)

with open(POST_LIST_FILE_PATH, 'w') as file:
    for title, path, mtime in posts:
        file.write('- [{} {}]({})\n\n'.format(time.strftime('%Y-%m-%d', mtime), title, md2url(path)))

rss2items = []

for title, path, mtime in posts:
    rss2items.append(PyRSS2Gen.RSSItem(
        title = title,
        link = '{}{}'.format(ROOT_URL, md2url(path)),
        guid = PyRSS2Gen.Guid(path),
        pubDate = datetime.datetime.fromtimestamp(time.mktime(mtime)) - datetime.timedelta(hours=8),
    ))

rss = PyRSS2Gen.RSS2(
    title = 'blog.so-link.org',
    link = 'https://blog.so-link.org/',
    items = rss2items,
    description = 'solink blog',
    lastBuildDate = datetime.datetime.now(),
)

rss_content = rss.to_xml(encoding='utf-8')
with open(RSS_XML_PATH, 'w') as file:
    file.write(rss_content)


"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import datetime
import dominate
from dominate.tags import *
import os


class HTML:
    def __init__(self, web_dir, title, refresh=0):
        if web_dir.endswith('.html'):
            web_dir, html_name = os.path.split(web_dir)
        else:
            web_dir, html_name = web_dir, 'index.html'
        self.title = title
        self.web_dir = web_dir
        self.html_name = html_name
        self.img_dir = os.path.join(self.web_dir, 'images')
        if len(self.web_dir) > 0 and not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if len(self.web_dir) > 0 and not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        with self.doc:
            pass
            # h1(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_hr(self):
        with self.doc:
            hr()

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=512):
        self.add_table()
        if isinstance(width, int):
            width = [width] * len(ims)
        with self.t:
            with tr():
                for im, txt, link, w in zip(ims, txts, links, width):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            p(txt, style="text-align: center;")
                            # br()
                            with a(href=os.path.join('images', link)):
                                img(loading="lazy", style="width:%dpx" % (w), src=os.path.join('images', im))

    def add_images_two_rows(self, ims, txts, links, num_cols,
                            empty_index=None, width=512):
        self.add_table()
        col_width = str(int(100/num_cols))
        with self.t:
            for start_index in range(0, len(ims), num_cols):
                end_index = min(num_cols+start_index, len(ims))
                with tr(): # text row
                    for i in range(start_index, end_index):
                        if i == empty_index:
                            with td(style=f"word-wrap: break-word; width: {col_width}%;", halign="center", valign="top"):
                                p()
                                # p("&nbsp;", style="text-align: center;")

                        with td(style=f"word-wrap: break-word; width: {col_width}%;", halign="center", valign="top"):
                            p(txts[i], style="text-align: center;")
                with tr(): # image row
                    for i in range(start_index, end_index):
                        if i == empty_index:
                            with td(style=f"word-wrap: break-word; width: {col_width}%;", halign="center", valign="top"):
                                p()
                                # p("&nbsp;", style="text-align: center;")
                        with td(style=f"word-wrap: break-word; width: {col_width}%;", halign="center", valign="top"):
                            with p(style="text-align: center;"):
                                # br()
                                with a(href=os.path.join('images', links[i])):
                                    img(loading="lazy", style="width:%dpx" % (width),
                                        src=os.path.join('images', ims[i]))
            br()


    def save(self):
        html_file = os.path.join(self.web_dir, self.html_name)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.jpg' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.jpg' % n)
    html.add_images(ims, txts, links)
    html.save()

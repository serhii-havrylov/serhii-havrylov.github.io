---
layout: page
title: Blog
order_id: 1
permalink: /blog.html
---

{% for post in site.posts %}
  * {{ post.date | date_to_string }} [ {{ post.title }} ]({{ post.url }})
{% endfor %}
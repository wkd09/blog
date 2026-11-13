---
title: "카테고리"
permalink: /categories/
layout: single
author_profile: true
---

{% assign study_count = site.categories.study.size | default: 0 %}
{% assign engineering_count = site.categories.engineering.size | default: 0 %}
{% assign research_count = site.categories.research.size | default: 0 %}
{% assign study_posts = site.categories.study %}
{% assign engineering_posts = site.categories.engineering %}
{% assign research_posts = site.categories.research %}

<section class="blog-intro">
  <p>글의 목적에 따라 공부, 구현, 논문 3가지 카테고리로 나눠 정리합니다.</p>
  <span>{{ site.posts.size }}개의 글</span>
</section>

<nav class="category-overview" aria-label="카테고리 바로가기">
  <a class="category-pill category-pill--study" href="#study">Study <span>{{ study_count }}</span></a>
  <a class="category-pill category-pill--engineering" href="#engineering">Engineering <span>{{ engineering_count }}</span></a>
  <a class="category-pill category-pill--research" href="#research">Research <span>{{ research_count }}</span></a>
</nav>

<section id="study" class="category-section">
  <div class="category-section__heading">
    <h2>Study</h2>
    <span>{{ study_count }}개의 글</span>
  </div>

  {% if study_posts.size > 0 %}
  <div class="post-list">
    {% for post in study_posts %}
      {% include post-list-item.html post=post %}
    {% endfor %}
  </div>
  {% else %}
  <div class="empty-state">
    <p>아직 작성한 공부 글이 없습니다.</p>
    <span>공부 기록을 추가하면 이곳에 표시됩니다.</span>
  </div>
  {% endif %}
</section>

<section id="engineering" class="category-section">
  <div class="category-section__heading">
    <h2>Engineering</h2>
    <span>{{ engineering_count }}개의 글</span>
  </div>

  {% if engineering_posts.size > 0 %}
  <div class="post-list">
    {% for post in engineering_posts %}
      {% include post-list-item.html post=post %}
    {% endfor %}
  </div>
  {% else %}
  <div class="empty-state">
    <p>아직 작성한 구현 글이 없습니다.</p>
    <span>구현 기록을 추가하면 이곳에 표시됩니다.</span>
  </div>
  {% endif %}
</section>

<section id="research" class="category-section">
  <div class="category-section__heading">
    <h2>Research</h2>
    <span>{{ research_count }}개의 글</span>
  </div>

  {% if research_posts.size > 0 %}
  <div class="post-list">
    {% for post in research_posts %}
      {% include post-list-item.html post=post %}
    {% endfor %}
  </div>
  {% else %}
  <div class="empty-state">
    <p>아직 작성한 논문 글이 없습니다.</p>
    <span>논문 리뷰를 추가하면 이곳에 표시됩니다.</span>
  </div>
  {% endif %}
</section>

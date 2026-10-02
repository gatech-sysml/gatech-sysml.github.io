---
title: Team
nav:
  order: 3
  tooltip: About our team
---

# {% include icon.html icon="fa-solid fa-users" %}Team

{% include section.html %}

## Principal Investigator
{% include list.html data="members" component="portrait" filter="role == 'pi'" %}

## PhD
{% include list.html data="members" component="portrait" filter="role == 'phd'" %}

## Masters
{% include list.html data="members" component="portrait" filter="role == 'masters'" %}

## Undergraduate
{% include list.html data="members" component="portrait" filter="role == 'undergrad'" %}

## Alumni
{% include list.html data="members" component="portrait" filter="role == 'phd-alumni'" %}
{% include list.html data="members" component="portrait" filter="role == 'masters-alumni'" %}
{% include list.html data="members" component="portrait" filter="role == 'undergrad-alumni'" %}

{% include section.html background="images/background.jpeg" dark=true %}

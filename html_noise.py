from __future__ import annotations

import random

from faker import Faker


fake = Faker()


def _attrs() -> dict[str, str]:
    """Return a small bundle of realistic-looking HTML attributes."""
    domain = fake.domain_name()
    return {
        "class": fake.word(),
        "id": fake.slug(),
        "href": f"https://{domain}/",
        "src": f"https://{domain}/{fake.slug()}.png",
        "content": fake.word(),
        "name": fake.word(),
        "property": random.choice(["og:title", "og:description", "og:image", "og:type"]),
        "lang": random.choice(["en", "fr", "de", "es", "pl", "pt", "zh", "ja"]),
    }


def generate_html_artifact() -> str:
    """Generate an HTML snippet with tags preserved and text content removed."""
    a = _attrs()

    templates = [
        (
            f'<!DOCTYPE html><html lang="{a["lang"]}"><head>'
            f'<meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title></title>'
            f'<meta name="description" content="">'
            f'<meta property="{a["property"]}" content="{a["content"]}">'
            f'<link rel="canonical" href="{a["href"]}">'
            f'</head><body><div class="{a["class"]}" id="{a["id"]}"><span></span></div></body></html>'
        ),
        (
            f'<header class="{a["class"]}"><nav><ul>'
            f'<li></li><li></li><li></li>'
            f'</ul></nav></header>'
        ),
        (
            f'<article id="{a["id"]}"><section class="{a["class"]}">'
            f'<h1></h1><p></p><p></p><aside></aside>'
            f'</section></article>'
        ),
        (
            f'<!-- {fake.slug()} --><script type="application/ld+json">{{}}</script>'
            f'<style></style><noscript></noscript>'
        ),
        (
            f'<div class="{a["class"]}"><div><span></span><span></span></div>'
            f'<footer><a href="{a["href"]}"></a></footer></div>'
        ),
        (
            f'<meta charset="utf-8"><meta name="{a["name"]}" content="{a["content"]}">'
            f'<link rel="preload" as="image" href="{a["src"]}">'
        ),
    ]

    return random.choice(templates)


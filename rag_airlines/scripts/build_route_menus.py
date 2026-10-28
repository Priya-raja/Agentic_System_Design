"""Reproducible fictional route-menu supplement; never real JetBlue policy."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import simpleSplit

ROOT = Path(__file__).resolve().parents[1]
ROUTES = [
    ('DXB', 'LHR', 'Dubai', 'London Heathrow', [
        ('Warm meal', 'Saffron chicken rice with roasted vegetables', 'AED 42 in Economy; included in Business'),
        ('Vegetarian meal', 'Chickpea and vegetable pilaf', 'AED 38; preorder recommended'),
        ('Light bite', 'Cheese and tomato wrap', 'AED 24'),
        ('Dessert', 'Date and orange cake', 'AED 18'),
        ('Drinks', 'Water AED 6; orange juice AED 12; tea or coffee AED 10', 'Still water included with an included Business meal'),
    ]),
    ('LHR', 'DXB', 'London Heathrow', 'Dubai', [
        ('Warm meal', 'Herb chicken with mashed potatoes and green beans', 'AED 44 in Economy; included in Business'),
        ('Vegetarian meal', 'Spinach and ricotta pasta', 'AED 39; preorder recommended'),
        ('Light bite', 'Roasted vegetable sandwich', 'AED 26'),
        ('Dessert', 'Apple and cinnamon crumble', 'AED 20'),
        ('Drinks', 'Water AED 6; apple juice AED 12; tea or coffee AED 10', 'Still water included with an included Business meal'),
    ]),
    ('DXB', 'BOM', 'Dubai', 'Mumbai', [
        ('Warm meal', 'Chicken tikka with jeera rice', 'AED 36 in Economy; included in Business'),
        ('Vegetarian meal', 'Chana masala with basmati rice', 'AED 32; preorder recommended'),
        ('Light bite', 'Paneer and pepper wrap', 'AED 22'),
        ('Dessert', 'Cardamom rice pudding', 'AED 16'),
        ('Drinks', 'Water AED 6; mango juice AED 12; masala tea AED 10', 'Still water included with an included Business meal'),
    ]),
    ('BOM', 'DXB', 'Mumbai', 'Dubai', [
        ('Warm meal', 'Chicken curry with steamed rice', 'AED 37 in Economy; included in Business'),
        ('Vegetarian meal', 'Vegetable biryani', 'AED 33; preorder recommended'),
        ('Light bite', 'Potato and pea pastry', 'AED 20'),
        ('Dessert', 'Mango yogurt pot', 'AED 15'),
        ('Drinks', 'Water AED 6; pineapple juice AED 12; tea or coffee AED 10', 'Still water included with an included Business meal'),
    ]),
    ('DXB', 'SIN', 'Dubai', 'Singapore', [
        ('Warm meal', 'Ginger chicken with jasmine rice', 'AED 46 in Economy; included in Business'),
        ('Vegetarian meal', 'Tofu and vegetable noodles', 'AED 40; preorder recommended'),
        ('Light bite', 'Mushroom and sesame bun', 'AED 25'),
        ('Dessert', 'Coconut and mango pudding', 'AED 19'),
        ('Drinks', 'Water AED 6; guava juice AED 13; tea or coffee AED 10', 'Still water included with an included Business meal'),
    ]),
    ('SIN', 'DXB', 'Singapore', 'Dubai', [
        ('Warm meal', 'Soy-glazed chicken with rice and greens', 'AED 45 in Economy; included in Business'),
        ('Vegetarian meal', 'Vegetable coconut curry with rice', 'AED 40; preorder recommended'),
        ('Light bite', 'Cucumber and hummus roll', 'AED 23'),
        ('Dessert', 'Pandan sponge cake', 'AED 18'),
        ('Drinks', 'Water AED 6; lychee juice AED 13; tea or coffee AED 10', 'Still water included with an included Business meal'),
    ]),
]


def build(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=(595.28, 841.89), invariant=True)
    c.setTitle('AeroNova | Directional Route Menus 2026')
    c.setAuthor('AeroNova fictional RAG demonstration')
    for page, (origin, dest, origin_name, dest_name, items) in enumerate(ROUTES, 1):
        c.setFillColor(HexColor('#F5F8FC')); c.rect(0, 0, 596, 842, fill=1, stroke=0)
        c.setFillColor(HexColor('#082A50')); c.rect(0, 686, 596, 156, fill=1, stroke=0)
        c.setFillColor(HexColor('#65D6EC')); c.setFont('Helvetica-Bold', 10)
        c.drawString(42, 803, 'AERONOVA AIRWAYS  /  FICTIONAL MENU COLLECTION')
        c.setFillColor(white); c.setFont('Helvetica-Bold', 28)
        c.drawString(42, 758, f'{origin} to {dest}')
        c.setFont('Helvetica', 12); c.drawString(42, 734, f'{origin_name} to {dest_name}')
        c.setFont('Helvetica', 9); c.drawString(42, 708, f'Route: {origin} -> {dest}  |  Direction-specific sample menu')
        c.setFillColor(HexColor('#465870')); c.setFont('Helvetica', 9)
        c.drawString(42, 665, 'CAT-CABIN-ROUTES-2026 | Version: 1.0 | Status: active')
        c.drawString(42, 650, 'Effective: 2026-01-01 to 2026-12-31 | All prices are illustrative AED')
        y = 614
        for category, name, price in items:
            c.setFillColor(HexColor('#007D99')); c.setFont('Helvetica-Bold', 9)
            c.drawString(42, y, category.upper())
            c.setFillColor(HexColor('#082A50')); c.setFont('Helvetica-Bold', 12)
            for line in simpleSplit(name, 'Helvetica-Bold', 12, 510):
                y -= 19; c.drawString(42, y, line)
            c.setFillColor(HexColor('#465870')); c.setFont('Helvetica', 10)
            for line in simpleSplit(price, 'Helvetica', 10, 510):
                y -= 17; c.drawString(42, y, line)
            y -= 16; c.setStrokeColor(HexColor('#D7E4EF')); c.line(42, y, 553, y); y -= 23
        c.setFillColor(HexColor('#E4F3F7')); c.roundRect(42, 101, 511, 108, 8, fill=1, stroke=0)
        c.setFillColor(HexColor('#082A50')); c.setFont('Helvetica-Bold', 10)
        c.drawString(56, 188, 'PLEASE NOTE')
        notes = [
            f'This sample applies only to {origin} to {dest}, not the reverse direction.',
            'Selections and prices are illustrative, subject to stock and aircraft service.',
            'Request special dietary meals at least 24 hours before departure;',
            'availability is not guaranteed. Ask cabin crew about allergens.',
            'This fictional supplement is not a real airline or JetBlue menu.',
        ]
        c.setFont('Helvetica', 9)
        for i, line in enumerate(notes): c.drawString(56, 171-i*13, line)
        c.setFillColor(HexColor('#66778B')); c.setFont('Helvetica', 8)
        c.drawString(42, 66, 'FICTIONAL TEST MATERIAL  |  No real airline policy')
        c.drawRightString(553, 66, f'{page} / {len(ROUTES)}')
        c.showPage()
    c.save()


if __name__ == '__main__':
    final = ROOT / 'data/policies/CAT-CABIN-ROUTES-2026.pdf'
    build(final)
    print(final)

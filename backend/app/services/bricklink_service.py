from typing import List, Dict, Any
from app.schemas.inventory import MissingPart

bricklink_color_map = {
    1: 16,
    2: 9,
    3: 20,
    4: 1,
    5: 2,
    6: 11,
    7: 12,
    8: 14,
    9: 13,
    10: 15,
    11: 18,
    12: 19,
    13: 21,
    14: 3,
    15: 4,
    16: 5,
    17: 6,
    18: 7,
    19: 8,
    21: 22,
    22: 10,
    23: 24,
    24: 25,
    25: 26,
    26: 27,
    27: 28,
    28: 29,
    29: 30,
    30: 31,
    33: 32,
    34: 33,
    35: 34,
    36: 35,
    37: 36,
    38: 37,
    40: 38,
    41: 39,
    42: 40,
    43: 41,
    44: 42,
    45: 43,
    46: 44,
    47: 45,
    48: 46,
    49: 47,
    50: 48,
}

bricklink_part_map_overrides = {
}


def generate_wanted_list_xml(missing_parts: List[MissingPart], condition: str = "N") -> str:
    xml_items = []

    for part in missing_parts:
        part_num = part.part_num
        if part_num in bricklink_part_map_overrides:
            part_num = bricklink_part_map_overrides[part_num]

        quantity_needed = part.quantity_needed - part.quantity_have
        if quantity_needed <= 0:
            continue

        color_id = 16
        if part.color_hex:
            for rb_color, bl_color in bricklink_color_map.items():
                if str(rb_color) == part.color_name or str(rb_color) == str(part.color_id):
                    color_id = bl_color
                    break

        xml_item = f"""    <ITEM>
        <ITEMID>{part_num}</ITEMID>
        <ITEMTYPE>P</ITEMTYPE>
        <COLOR>{color_id}</COLOR>
        <MINQTY>{quantity_needed}</MINQTY>
        <CONDITION>{condition}</CONDITION>
    </ITEM>
"""
        xml_items.append(xml_item)

    items_xml = "\n".join(xml_items)

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<INVENTORY>
{items_xml}
</INVENTORY>
"""

    return xml

"""TLC Taxi Zone location ID to borough mapping.

Maps ~265 taxi zone IDs to their corresponding NYC borough.
Source: NYC TLC Taxi Zone Lookup Table.
"""

# Borough mapping for all TLC taxi zones (LocationID -> Borough)
# Zones 1-263 are valid taxi zones; 264/265 are unknown/NV
LOCATION_TO_BOROUGH: dict[int, str] = {
    1: "EWR", 2: "Queens", 3: "Bronx", 4: "Manhattan", 5: "Staten Island",
    6: "Staten Island", 7: "Queens", 8: "Queens", 9: "Queens", 10: "Queens",
    11: "Queens", 12: "Manhattan", 13: "Manhattan", 14: "Brooklyn", 15: "Queens",
    16: "Queens", 17: "Brooklyn", 18: "Bronx", 19: "Queens", 20: "Queens",
    21: "Manhattan", 22: "Queens", 23: "Queens", 24: "Manhattan", 25: "Manhattan",
    26: "Brooklyn", 27: "Brooklyn", 28: "Queens", 29: "Queens", 30: "Queens",
    31: "Brooklyn", 32: "Brooklyn", 33: "Brooklyn", 34: "Manhattan", 35: "Brooklyn",
    36: "Brooklyn", 37: "Brooklyn", 38: "Queens", 39: "Brooklyn", 40: "Bronx",
    41: "Manhattan", 42: "Manhattan", 43: "Manhattan", 44: "Manhattan", 45: "Manhattan",
    46: "Bronx", 47: "Manhattan", 48: "Manhattan", 49: "Brooklyn", 50: "Manhattan",
    51: "Manhattan", 52: "Manhattan", 53: "Brooklyn", 54: "Brooklyn", 55: "Brooklyn",
    56: "Brooklyn", 57: "Brooklyn", 58: "Brooklyn", 59: "Queens", 60: "Brooklyn",
    61: "Brooklyn", 62: "Brooklyn", 63: "Brooklyn", 64: "Manhattan", 65: "Brooklyn",
    66: "Manhattan", 67: "Brooklyn", 68: "Manhattan", 69: "Brooklyn", 70: "Brooklyn",
    71: "Brooklyn", 72: "Brooklyn", 73: "Queens", 74: "Manhattan", 75: "Manhattan",
    76: "Brooklyn", 77: "Brooklyn", 78: "Brooklyn", 79: "Manhattan", 80: "Manhattan",
    81: "Brooklyn", 82: "Queens", 83: "Queens", 84: "Manhattan", 85: "Queens",
    86: "Manhattan", 87: "Manhattan", 88: "Manhattan", 89: "Brooklyn", 90: "Manhattan",
    91: "Queens", 92: "Queens", 93: "Queens", 94: "Brooklyn", 95: "Queens",
    96: "Queens", 97: "Brooklyn", 98: "Queens", 99: "Queens", 100: "Manhattan",
    101: "Queens", 102: "Manhattan", 103: "Manhattan", 104: "Manhattan", 105: "Brooklyn",
    106: "Brooklyn", 107: "Manhattan", 108: "Manhattan", 109: "Brooklyn", 110: "Brooklyn",
    111: "Manhattan", 112: "Manhattan", 113: "Manhattan", 114: "Manhattan", 115: "Queens",
    116: "Manhattan", 117: "Brooklyn", 118: "Brooklyn", 119: "Brooklyn", 120: "Queens",
    121: "Manhattan", 122: "Queens", 123: "Manhattan", 124: "Manhattan", 125: "Manhattan",
    126: "Queens", 127: "Queens", 128: "Manhattan", 129: "Queens", 130: "Manhattan",
    131: "Queens", 132: "Queens", 133: "Brooklyn", 134: "Queens", 135: "Queens",
    136: "Queens", 137: "Queens", 138: "Queens", 139: "Queens", 140: "Manhattan",
    141: "Manhattan", 142: "Manhattan", 143: "Manhattan", 144: "Manhattan", 145: "Manhattan",
    146: "Queens", 147: "Bronx", 148: "Manhattan", 149: "Brooklyn", 150: "Brooklyn",
    151: "Manhattan", 152: "Manhattan", 153: "Manhattan", 154: "Brooklyn", 155: "Brooklyn",
    156: "Manhattan", 157: "Manhattan", 158: "Manhattan", 159: "Queens", 160: "Manhattan",
    161: "Manhattan", 162: "Manhattan", 163: "Manhattan", 164: "Manhattan", 165: "Manhattan",
    166: "Manhattan", 167: "Brooklyn", 168: "Manhattan", 169: "Brooklyn", 170: "Manhattan",
    171: "Brooklyn", 172: "Brooklyn", 173: "Manhattan", 174: "Brooklyn", 175: "Brooklyn",
    176: "Brooklyn", 177: "Brooklyn", 178: "Queens", 179: "Manhattan", 180: "Queens",
    181: "Queens", 182: "Queens", 183: "Queens", 184: "Brooklyn", 185: "Brooklyn",
    186: "Manhattan", 187: "Brooklyn", 188: "Manhattan", 189: "Brooklyn", 190: "Brooklyn",
    191: "Manhattan", 192: "Queens", 193: "Queens", 194: "Queens", 195: "Queens",
    196: "Queens", 197: "Brooklyn", 198: "Brooklyn", 199: "Brooklyn", 200: "Manhattan",
    201: "Brooklyn", 202: "Manhattan", 203: "Manhattan", 204: "Manhattan", 205: "Queens",
    206: "Queens", 207: "Queens", 208: "Queens", 209: "Manhattan", 210: "Queens",
    211: "Manhattan", 212: "Manhattan", 213: "Manhattan", 214: "Brooklyn", 215: "Brooklyn",
    216: "Brooklyn", 217: "Brooklyn", 218: "Brooklyn", 219: "Brooklyn", 220: "Brooklyn",
    221: "Queens", 222: "Queens", 223: "Manhattan", 224: "Manhattan", 225: "Brooklyn",
    226: "Brooklyn", 227: "Manhattan", 228: "Manhattan", 229: "Manhattan", 230: "Manhattan",
    231: "Manhattan", 232: "Manhattan", 233: "Manhattan", 234: "Manhattan", 235: "Manhattan",
    236: "Manhattan", 237: "Manhattan", 238: "Manhattan", 239: "Manhattan", 240: "Manhattan",
    241: "Manhattan", 242: "Manhattan", 243: "Manhattan", 244: "Manhattan", 245: "Manhattan",
    246: "Brooklyn", 247: "Brooklyn", 248: "Manhattan", 249: "Manhattan", 250: "Brooklyn",
    251: "Manhattan", 252: "Queens", 253: "Manhattan", 254: "Brooklyn", 255: "Brooklyn",
    256: "Brooklyn", 257: "Queens", 258: "Queens", 259: "Queens", 260: "Queens",
    261: "Manhattan", 262: "Manhattan", 263: "Manhattan",
}

# Simplified borough set for validation
VALID_BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]


def get_borough(location_id: int) -> str:
    """Get borough name for a TLC location ID.

    Args:
        location_id: TLC taxi zone location ID.

    Returns:
        Borough name, or 'Unknown' if not found.
    """
    return LOCATION_TO_BOROUGH.get(location_id, "Unknown")

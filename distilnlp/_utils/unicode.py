import unicodedata

# unicode category is 'Zs'
space_separators = ('\u0020', '\u00A0', '\u1680', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200A', '\u202F', '\u205F', '\u3000')

# space_separators + \f\n\r\t\v
space_symbol = space_separators + ('\f', '\n', '\r', '\t', '\v')

def is_printable_symbol(ch):
    if not ch.isprintable():
        return False
    
    category = unicodedata.category(ch)
    if category in ('Mn', 'Co'): # Non-spacing Mark, Private Use
        return False
    
    return True

def is_exceptional_symbol(ch):
    '''Excluding Non-Printable Characters'''

    # https://en.wikipedia.org/wiki/Specials_(Unicode_block)
    if ch in ('\uFFFC', '\uFFFD'):
        return True
    return False

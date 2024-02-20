import unicodedata

def is_printable_symbol(ch):
    if not ch.isprintable():
        return False
    
    category = unicodedata.category(ch)
    if category in ('Mn', 'Co'): # Non-spacing Mark, Private Use
        return False
    
    return True
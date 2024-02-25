import unicodedata

def is_printable_symbol(ch):
    if not ch.isprintable():
        return False
    
    category = unicodedata.category(ch)
    if category in ('Mn', ): # Non-spacing Mark
        return False
    
    return True
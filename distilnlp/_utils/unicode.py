import unicodedata

def is_mystery_symbol(ch):
    if not ch.isprintable() and not ch in ('\n', '\t'):
        return True
    
    category = unicodedata.category(ch)
    if category in ('Mn', ): # Non-spacing Mark
        return True
    
    return False
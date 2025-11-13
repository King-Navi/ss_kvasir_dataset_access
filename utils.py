def build_class_mappings():
    """
    Build mappings:
      - full_class_name -> short_key (e.g. 'Polyp' -> 'polyp')
      - full_class_name -> numeric_id (0..13)
    Based directly on FINDING_CLASS_MAP insertion order.
    """
    full_to_key = {v: k for k, v in FINDING_CLASS_MAP.items()}
    full_to_id = {v: idx for idx, (k, v) in enumerate(FINDING_CLASS_MAP.items())}
    return full_to_key, full_to_id
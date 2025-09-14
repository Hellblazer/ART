# ASCII Diagram Templates

## Box Drawing Guide

For consistent ASCII boxes, use these templates:

### Standard Box (61 chars wide)
```
+-------------------------------------------------------------+
|                         Title Text                         |
+-------------------------------------------------------------+
```

### Box with Content
```
+-------------------------------------------------------------+
|                         Title Text                         |
|  * Bullet point 1                                          |
|  * Bullet point 2                                          |
|  * Bullet point 3                                          |
+-------------------------------------------------------------+
```

### Box with Arrow Down
```
+-------------------------------------------------------------+
|                         Title Text                         |
+-----------------------------+-------------------------------+
                              |
                              v
```

### Simple Alternative (using regular ASCII)
```
+-------------------------------------------------------------+
|                         Input Text                         |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                         Tokenizer                          |
|  * Edge case handling                                      |
|  * Type classification                                     |
|  * Timestamp tracking                                      |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                 Sliding Window (20 tokens)                 |
|  * 5-token overlap                                         |
|  * Boundary detection                                      |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|            Multi-Channel Processor (6 channels)            |
|  +----------+ +----------+ +----------+ +----------+       |
|  |Positional| |   Word   | | Context  | |Structural| ...   |
|  +----------+ +----------+ +----------+ +----------+       |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|           Hierarchical Processor (3 levels)                |
|  * Level 1: Morpheme (p=0.9)                               |
|  * Level 2: Phrase (p=0.7)                                 |
|  * Level 3: Discourse (p=0.5)                              |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|            Competitive Queue (Grossberg)                   |
|  * Self-excitation: 1.2                                    |
|  * Lateral inhibition: 0.3                                 |
|  * Winner-take-all selection                               |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    Template Selection                      |
|  * 25+ predefined templates                                |
|  * Slot-based filling                                      |
|  * Deterministic output                                    |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                        Output Text                         |
+-------------------------------------------------------------+
```

## Tips for Clean ASCII Diagrams

1. **Use fixed width**: Pick a standard width (e.g., 61 chars) and stick to it
2. **Center titles**: Calculate padding for centered text
3. **Align bullets**: Use consistent spacing for bullet points
4. **Simple characters**: Use `+`, `-`, `|` for better compatibility
5. **Arrows**: Use `v` or `V` for down arrows, `>` for right arrows

## Character Reference

### Box Drawing (Unicode)
- Corners: `┌` `┐` `└` `┘`
- Lines: `─` `│`
- Junctions: `├` `┤` `┬` `┴` `┼`

### Simple ASCII
- Corners: `+`
- Horizontal: `-`
- Vertical: `|`
- Arrows: `v` `^` `<` `>`

## Width Calculation

For a 61-character wide box:
- Total width: 61 chars
- Border chars: 2 (left `|` and right `|`)
- Content area: 59 chars
- Text centering: `(59 - text_length) / 2` spaces on each side

Example centering "Input Text" (10 chars):
- Padding needed: (59 - 10) / 2 = 24.5
- Use 24 spaces before and 25 after (or vice versa)
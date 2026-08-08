# Сколько 2-tok слов с перекосом attention от «чужих» токенов

Traces: **64**, layer: **16**

Per-word share = attn(other→subword0) / (attn→subword0 + attn→subword1).
0.5 = симметрия; перекос = share заметно отличается от 0.5.

## Фаза `before_first`

### По позиции (subword0 vs subword1) (n=109)

- mean share: **0.507** (0.5 = symmetric)
- mean |share−0.5|: **0.101**
- median |share−0.5|: **0.092**
- |share−0.5| > 0.05: **74.3%** слов
- |share−0.5| > 0.10: **45.9%** слов
- |share−0.5| > 0.15: **25.7%** слов
- |share−0.5| > 0.20: **10.1%** слов

## Фаза `between`

### По позиции (subword0 vs subword1) (n=109)

- mean share: **0.458** (0.5 = symmetric)
- mean |share−0.5|: **0.155**
- median |share−0.5|: **0.143**
- |share−0.5| > 0.05: **78.0%** слов
- |share−0.5| > 0.10: **63.3%** слов
- |share−0.5| > 0.15: **48.6%** слов
- |share−0.5| > 0.20: **34.9%** слов

### По состоянию (open vs masked subword) (n=109)

- mean share: **0.489** (0.5 = symmetric)
- mean |share−0.5|: **0.155**
- median |share−0.5|: **0.143**
- |share−0.5| > 0.05: **78.0%** слов
- |share−0.5| > 0.10: **63.3%** слов
- |share−0.5| > 0.15: **48.6%** слов
- |share−0.5| > 0.20: **34.9%** слов

## Фаза `all_open`

### По позиции (subword0 vs subword1) (n=109)

- mean share: **0.423** (0.5 = symmetric)
- mean |share−0.5|: **0.137**
- median |share−0.5|: **0.129**
- |share−0.5| > 0.05: **75.2%** слов
- |share−0.5| > 0.10: **56.9%** слов
- |share−0.5| > 0.15: **45.0%** слов
- |share−0.5| > 0.20: **26.6%** слов

## Примеры сильного перекоса (|pos_share−0.5| > 0.15)

- `seabed` (between): |Δ|=0.41
- `AIF` (all_open): |Δ|=0.40
- `Boer` (between): |Δ|=0.39
- `Somme` (between): |Δ|=0.38
- `Advancing` (all_open): |Δ|=0.36
- `NHC` (between): |Δ|=0.36
- `Ancre` (all_open): |Δ|=0.35
- `grossing` (between): |Δ|=0.35
- `Tikal` (between): |Δ|=0.35
- `Tikal` (all_open): |Δ|=0.33
- `Somme` (all_open): |Δ|=0.33
- `pascal` (between): |Δ|=0.32
- `garrison` (between): |Δ|=0.32
- `battleships` (all_open): |Δ|=0.31
- `Harare` (before_first): |Δ|=0.31

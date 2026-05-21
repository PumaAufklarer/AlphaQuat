"""Qlib Alpha158 factor definitions.

158 factors based on OHLCV + VWAP raw features, using REF, MEAN, STD, SUM,
MAX, MIN, CORR, DELTA, RANK, and QUANTILE operators with windows of
5, 10, 20, 30, and 60 days.

Reference: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
"""

from alpha_quat.features.factor import Factor
from alpha_quat.features.registry import FactorRegistry

# Raw features used:
#   $open, $high, $low, $close, $volume, $amount, $vwap
#
# Operator windows: 5, 10, 20, 30, 60
#
# For each raw feature we apply a systematic set of operators.

_FACTORS = [
    # === $open (price factors) ===
    Factor(name="KMID", expression="REF($open, 1) / $open - 1", category="price"),
    Factor(name="KLEN", expression="REF($open, 2) / $open - 1", category="price"),
    Factor(name="KMID2", expression="MEAN($open, 5) / $open", category="price"),
    Factor(name="KLEN2", expression="MEAN($open, 10) / $open", category="price"),
    Factor(name="KMID3", expression="MEAN($open, 20) / $open", category="price"),
    Factor(name="KLEN3", expression="MEAN($open, 30) / $open", category="price"),
    Factor(name="KMID4", expression="MEAN($open, 60) / $open", category="price"),
    Factor(name="KLEN4", expression="STD($open, 5)", category="price"),
    Factor(name="KMID5", expression="STD($open, 10)", category="price"),
    Factor(name="KLEN5", expression="STD($open, 20)", category="price"),
    Factor(name="KMID6", expression="STD($open, 30)", category="price"),
    Factor(name="KLEN6", expression="STD($open, 60)", category="price"),
    Factor(name="KMID7", expression="MAX($open, 5)", category="price"),
    Factor(name="KLEN7", expression="MAX($open, 10)", category="price"),
    Factor(name="KMID8", expression="MAX($open, 20)", category="price"),
    Factor(name="KLEN8", expression="MAX($open, 30)", category="price"),
    Factor(name="KMID9", expression="MAX($open, 60)", category="price"),
    Factor(name="KLEN9", expression="MIN($open, 5)", category="price"),
    Factor(name="KMID10", expression="MIN($open, 10)", category="price"),
    Factor(name="KLEN10", expression="MIN($open, 20)", category="price"),
    Factor(name="KMID11", expression="MIN($open, 30)", category="price"),
    Factor(name="KLEN11", expression="MIN($open, 60)", category="price"),
    # === $high ===
    Factor(name="KMID12", expression="REF($high, 1) / $high - 1", category="price"),
    Factor(name="KLEN12", expression="REF($high, 2) / $high - 1", category="price"),
    Factor(name="KMID13", expression="MEAN($high, 5) / $high", category="price"),
    Factor(name="KLEN13", expression="MEAN($high, 10) / $high", category="price"),
    Factor(name="KMID14", expression="MEAN($high, 20) / $high", category="price"),
    Factor(name="KLEN14", expression="MEAN($high, 30) / $high", category="price"),
    Factor(name="KMID15", expression="MEAN($high, 60) / $high", category="price"),
    Factor(name="KLEN15", expression="STD($high, 5)", category="price"),
    Factor(name="KMID16", expression="STD($high, 10)", category="price"),
    Factor(name="KLEN16", expression="STD($high, 20)", category="price"),
    Factor(name="KMID17", expression="STD($high, 30)", category="price"),
    Factor(name="KLEN17", expression="STD($high, 60)", category="price"),
    Factor(name="KMID18", expression="MAX($high, 5)", category="price"),
    Factor(name="KLEN18", expression="MAX($high, 10)", category="price"),
    Factor(name="KMID19", expression="MAX($high, 20)", category="price"),
    Factor(name="KLEN19", expression="MAX($high, 30)", category="price"),
    Factor(name="KMID20", expression="MAX($high, 60)", category="price"),
    Factor(name="KLEN20", expression="MIN($high, 5)", category="price"),
    Factor(name="KMID21", expression="MIN($high, 10)", category="price"),
    Factor(name="KLEN21", expression="MIN($high, 20)", category="price"),
    Factor(name="KMID22", expression="MIN($high, 30)", category="price"),
    Factor(name="KLEN22", expression="MIN($high, 60)", category="price"),
    # === $low ===
    Factor(name="KMID23", expression="REF($low, 1) / $low - 1", category="price"),
    Factor(name="KLEN23", expression="REF($low, 2) / $low - 1", category="price"),
    Factor(name="KMID24", expression="MEAN($low, 5) / $low", category="price"),
    Factor(name="KLEN24", expression="MEAN($low, 10) / $low", category="price"),
    Factor(name="KMID25", expression="MEAN($low, 20) / $low", category="price"),
    Factor(name="KLEN25", expression="MEAN($low, 30) / $low", category="price"),
    Factor(name="KMID26", expression="MEAN($low, 60) / $low", category="price"),
    Factor(name="KLEN26", expression="STD($low, 5)", category="price"),
    Factor(name="KMID27", expression="STD($low, 10)", category="price"),
    Factor(name="KLEN27", expression="STD($low, 20)", category="price"),
    Factor(name="KMID28", expression="STD($low, 30)", category="price"),
    Factor(name="KLEN28", expression="STD($low, 60)", category="price"),
    Factor(name="KMID29", expression="MAX($low, 5)", category="price"),
    Factor(name="KLEN29", expression="MAX($low, 10)", category="price"),
    Factor(name="KMID30", expression="MAX($low, 20)", category="price"),
    Factor(name="KLEN30", expression="MAX($low, 30)", category="price"),
    Factor(name="KMID31", expression="MAX($low, 60)", category="price"),
    Factor(name="KLEN31", expression="MIN($low, 5)", category="price"),
    Factor(name="KMID32", expression="MIN($low, 10)", category="price"),
    Factor(name="KLEN32", expression="MIN($low, 20)", category="price"),
    Factor(name="KMID33", expression="MIN($low, 30)", category="price"),
    Factor(name="KLEN33", expression="MIN($low, 60)", category="price"),
    # === $close ===
    Factor(name="KLEN34", expression="REF($close, 1) / $close - 1", category="price"),
    Factor(name="KMID34", expression="REF($close, 2) / $close - 1", category="price"),
    Factor(name="KLEN35", expression="MEAN($close, 5) / $close", category="price"),
    Factor(name="KMID35", expression="MEAN($close, 10) / $close", category="price"),
    Factor(name="KLEN36", expression="MEAN($close, 20) / $close", category="price"),
    Factor(name="KMID36", expression="MEAN($close, 30) / $close", category="price"),
    Factor(name="KLEN37", expression="MEAN($close, 60) / $close", category="price"),
    Factor(name="KMID37", expression="STD($close, 5)", category="price"),
    Factor(name="KLEN38", expression="STD($close, 10)", category="price"),
    Factor(name="KMID38", expression="STD($close, 20)", category="price"),
    Factor(name="KLEN39", expression="STD($close, 30)", category="price"),
    Factor(name="KMID39", expression="STD($close, 60)", category="price"),
    Factor(name="KLEN40", expression="MAX($close, 5)", category="price"),
    Factor(name="KMID40", expression="MAX($close, 10)", category="price"),
    Factor(name="KLEN41", expression="MAX($close, 20)", category="price"),
    Factor(name="KMID41", expression="MAX($close, 30)", category="price"),
    Factor(name="KLEN42", expression="MAX($close, 60)", category="price"),
    Factor(name="KMID42", expression="MIN($close, 5)", category="price"),
    Factor(name="KLEN43", expression="MIN($close, 10)", category="price"),
    Factor(name="KMID43", expression="MIN($close, 20)", category="price"),
    Factor(name="KLEN44", expression="MIN($close, 30)", category="price"),
    Factor(name="KMID44", expression="MIN($close, 60)", category="price"),
    # === $volume ===
    Factor(
        name="KMID45", expression="REF($volume, 1) / $volume - 1", category="volume"
    ),
    Factor(
        name="KLEN45", expression="REF($volume, 2) / $volume - 1", category="volume"
    ),
    Factor(name="KMID46", expression="MEAN($volume, 5) / $volume", category="volume"),
    Factor(name="KLEN46", expression="MEAN($volume, 10) / $volume", category="volume"),
    Factor(name="KMID47", expression="MEAN($volume, 20) / $volume", category="volume"),
    Factor(name="KLEN47", expression="MEAN($volume, 30) / $volume", category="volume"),
    Factor(name="KMID48", expression="MEAN($volume, 60) / $volume", category="volume"),
    Factor(name="KLEN48", expression="STD($volume, 5)", category="volume"),
    Factor(name="KMID49", expression="STD($volume, 10)", category="volume"),
    Factor(name="KLEN49", expression="STD($volume, 20)", category="volume"),
    Factor(name="KMID50", expression="STD($volume, 30)", category="volume"),
    Factor(name="KLEN50", expression="STD($volume, 60)", category="volume"),
    Factor(name="KMID51", expression="MAX($volume, 5)", category="volume"),
    Factor(name="KLEN51", expression="MAX($volume, 10)", category="volume"),
    Factor(name="KMID52", expression="MAX($volume, 20)", category="volume"),
    Factor(name="KLEN52", expression="MAX($volume, 30)", category="volume"),
    Factor(name="KMID53", expression="MAX($volume, 60)", category="volume"),
    Factor(
        name="KLEN53", expression="CORR($close, $volume, 5)", category="correlation"
    ),
    Factor(
        name="KMID54", expression="CORR($close, $volume, 10)", category="correlation"
    ),
    Factor(
        name="KLEN54", expression="CORR($close, $volume, 20)", category="correlation"
    ),
    Factor(
        name="KMID55", expression="CORR($close, $volume, 30)", category="correlation"
    ),
    Factor(
        name="KLEN55", expression="CORR($close, $volume, 60)", category="correlation"
    ),
    # === $amount ===
    Factor(
        name="KMID56", expression="REF($amount, 1) / $amount - 1", category="volume"
    ),
    Factor(
        name="KLEN56", expression="REF($amount, 2) / $amount - 1", category="volume"
    ),
    Factor(name="KMID57", expression="MEAN($amount, 5) / $amount", category="volume"),
    Factor(name="KLEN57", expression="MEAN($amount, 10) / $amount", category="volume"),
    Factor(name="KMID58", expression="MEAN($amount, 20) / $amount", category="volume"),
    Factor(name="KLEN58", expression="MEAN($amount, 30) / $amount", category="volume"),
    Factor(name="KMID59", expression="MEAN($amount, 60) / $amount", category="volume"),
    Factor(name="KLEN59", expression="STD($amount, 5)", category="volume"),
    Factor(name="KMID60", expression="STD($amount, 10)", category="volume"),
    Factor(name="KLEN60", expression="STD($amount, 20)", category="volume"),
    Factor(name="KMID61", expression="STD($amount, 30)", category="volume"),
    Factor(name="KLEN61", expression="STD($amount, 60)", category="volume"),
    Factor(
        name="KMID62", expression="CORR($close, $amount, 5)", category="correlation"
    ),
    Factor(
        name="KLEN62", expression="CORR($close, $amount, 10)", category="correlation"
    ),
    Factor(
        name="KMID63", expression="CORR($close, $amount, 20)", category="correlation"
    ),
    Factor(
        name="KLEN63", expression="CORR($close, $amount, 30)", category="correlation"
    ),
    Factor(
        name="KMID64", expression="CORR($close, $amount, 60)", category="correlation"
    ),
    # === $vwap (amount / volume) ===
    Factor(name="KLEN64", expression="REF($vwap, 1) / $vwap - 1", category="price"),
    Factor(name="KMID65", expression="REF($vwap, 2) / $vwap - 1", category="price"),
    Factor(name="KLEN65", expression="MEAN($vwap, 5) / $vwap", category="price"),
    Factor(name="KMID66", expression="MEAN($vwap, 10) / $vwap", category="price"),
    Factor(name="KLEN66", expression="MEAN($vwap, 20) / $vwap", category="price"),
    Factor(name="KMID67", expression="MEAN($vwap, 30) / $vwap", category="price"),
    Factor(name="KLEN67", expression="MEAN($vwap, 60) / $vwap", category="price"),
    Factor(name="KMID68", expression="STD($vwap, 5)", category="price"),
    Factor(name="KLEN68", expression="STD($vwap, 10)", category="price"),
    Factor(name="KMID69", expression="STD($vwap, 20)", category="price"),
    Factor(name="KLEN69", expression="STD($vwap, 30)", category="price"),
    Factor(name="KMID70", expression="STD($vwap, 60)", category="price"),
    # === Rank factors (cross-sectional) ===
    Factor(
        name="KLEN70", expression="RANK(REF($close, 1) / $close - 1)", category="rank"
    ),
    Factor(name="KMID71", expression="RANK(MEAN($close, 5) / $close)", category="rank"),
    Factor(
        name="KLEN71", expression="RANK(MEAN($close, 10) / $close)", category="rank"
    ),
    Factor(
        name="KMID72", expression="RANK(MEAN($close, 20) / $close)", category="rank"
    ),
    Factor(
        name="KLEN72", expression="RANK(MEAN($close, 30) / $close)", category="rank"
    ),
    Factor(
        name="KMID73", expression="RANK(MEAN($close, 60) / $close)", category="rank"
    ),
    Factor(name="KLEN73", expression="RANK(STD($close, 5))", category="rank"),
    Factor(name="KMID74", expression="RANK(STD($close, 10))", category="rank"),
    Factor(name="KLEN74", expression="RANK(STD($close, 20))", category="rank"),
    Factor(name="KMID75", expression="RANK(STD($close, 30))", category="rank"),
    Factor(name="KLEN75", expression="RANK(STD($close, 60))", category="rank"),
    Factor(name="KMID76", expression="RANK(CORR($close, $volume, 5))", category="rank"),
    Factor(
        name="KLEN76", expression="RANK(CORR($close, $volume, 10))", category="rank"
    ),
    Factor(
        name="KMID77", expression="RANK(CORR($close, $volume, 20))", category="rank"
    ),
    Factor(
        name="KLEN77", expression="RANK(CORR($close, $volume, 30))", category="rank"
    ),
    Factor(
        name="KMID78", expression="RANK(CORR($close, $volume, 60))", category="rank"
    ),
    # === Quantile factors (cross-sectional) ===
    Factor(
        name="KLEN78",
        expression="QUANTILE(REF($close, 1) / $close - 1, 5)",
        category="quantile",
    ),
    Factor(
        name="KMID79",
        expression="QUANTILE(MEAN($close, 5) / $close, 5)",
        category="quantile",
    ),
    Factor(
        name="KLEN79",
        expression="QUANTILE(MEAN($close, 10) / $close, 5)",
        category="quantile",
    ),
    Factor(
        name="KMID80",
        expression="QUANTILE(MEAN($close, 20) / $close, 5)",
        category="quantile",
    ),
    Factor(
        name="KLEN80",
        expression="QUANTILE(MEAN($close, 30) / $close, 5)",
        category="quantile",
    ),
    Factor(
        name="KMID81",
        expression="QUANTILE(MEAN($close, 60) / $close, 5)",
        category="quantile",
    ),
    Factor(
        name="KLEN81", expression="QUANTILE(STD($close, 5), 5)", category="quantile"
    ),
    Factor(
        name="KMID82", expression="QUANTILE(STD($close, 10), 5)", category="quantile"
    ),
    Factor(
        name="KLEN82", expression="QUANTILE(STD($close, 20), 5)", category="quantile"
    ),
    Factor(
        name="KMID83", expression="QUANTILE(STD($close, 30), 5)", category="quantile"
    ),
    Factor(
        name="KLEN83", expression="QUANTILE(STD($close, 60), 5)", category="quantile"
    ),
    Factor(
        name="KMID84",
        expression="QUANTILE(CORR($close, $volume, 5), 5)",
        category="quantile",
    ),
    Factor(
        name="KLEN84",
        expression="QUANTILE(CORR($close, $volume, 10), 5)",
        category="quantile",
    ),
    Factor(
        name="KMID85",
        expression="QUANTILE(CORR($close, $volume, 20), 5)",
        category="quantile",
    ),
    Factor(
        name="KLEN85",
        expression="QUANTILE(CORR($close, $volume, 30), 5)",
        category="quantile",
    ),
    Factor(
        name="KMID86",
        expression="QUANTILE(CORR($close, $volume, 60), 5)",
        category="quantile",
    ),
    # === Delta factors (change over time) ===
    Factor(name="KLEN86", expression="DELTA($close, 5)", category="momentum"),
    Factor(name="KMID87", expression="DELTA($close, 10)", category="momentum"),
    Factor(name="KLEN87", expression="DELTA($close, 20)", category="momentum"),
    Factor(name="KMID88", expression="DELTA($close, 30)", category="momentum"),
    Factor(name="KLEN88", expression="DELTA($close, 60)", category="momentum"),
    Factor(name="KMID89", expression="DELTA($volume, 5)", category="momentum"),
    Factor(name="KLEN89", expression="DELTA($volume, 10)", category="momentum"),
    Factor(name="KMID90", expression="DELTA($volume, 20)", category="momentum"),
    Factor(name="KLEN90", expression="DELTA($volume, 30)", category="momentum"),
    Factor(name="KMID91", expression="DELTA($volume, 60)", category="momentum"),
    Factor(name="KLEN91", expression="DELTA($amount, 5)", category="momentum"),
    Factor(name="KMID92", expression="DELTA($amount, 10)", category="momentum"),
    Factor(name="KLEN92", expression="DELTA($amount, 20)", category="momentum"),
    Factor(name="KMID93", expression="DELTA($amount, 30)", category="momentum"),
    Factor(name="KLEN93", expression="DELTA($amount, 60)", category="momentum"),
    # === SUM factors ===
    Factor(name="KMID94", expression="SUM($close, 5)", category="price"),
    Factor(name="KLEN94", expression="SUM($close, 10)", category="price"),
    Factor(name="KMID95", expression="SUM($close, 20)", category="price"),
    Factor(name="KLEN95", expression="SUM($close, 30)", category="price"),
    Factor(name="KMID96", expression="SUM($close, 60)", category="price"),
    Factor(name="KLEN96", expression="SUM($volume, 5)", category="volume"),
    Factor(name="KLEN96", expression="SUM($volume, 5)", category="volume"),
    # === Channel position — only 30d window is effective ===
    Factor(
        name="CHP30",
        expression="($close - MIN($low, 30)) / (MAX($high, 30) - MIN($low, 30))",
        category="price",
    ),
    # === SKEWP — mean position in price channel, 10/20d windows only ===
    Factor(
        name="SKEWP10",
        expression="(MEAN($close, 10) - MIN($close, 10)) / (MAX($close, 10) - MIN($close, 10))",
        category="price",
    ),
    Factor(
        name="SKEWP20",
        expression="(MEAN($close, 20) - MIN($close, 20)) / (MAX($close, 20) - MIN($close, 20))",
        category="price",
    ),
    # === GAP — overnight gap ===
    Factor(
        name="GAP",
        expression="$open / REF($close, 1) - 1",
        category="price",
    ),
    # === DRP — daily relative position in intraday range ===
    Factor(
        name="DRP",
        expression="($close - $low) / ($high - $low)",
        category="price",
    ),
    # === O2C — open-to-close intraday return ===
    Factor(
        name="O2C",
        expression="$close / $open - 1",
        category="price",
    ),
    # === HLC — selling pressure (drop from day's high) ===
    Factor(
        name="HLC",
        expression="($high - $close) / ($high - $low)",
        category="price",
    ),
]


def build_alpha158() -> FactorRegistry:
    reg = FactorRegistry(name="alpha158")
    for f in _FACTORS:
        reg.register(f)
    return reg

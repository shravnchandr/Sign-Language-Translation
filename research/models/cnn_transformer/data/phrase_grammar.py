"""
Gloss-phrase grammar over the 250 ASL Signs (PopSign) vocabulary.

Each sample is (glosses, english): a short ASL-style gloss sequence and an
English rendering, e.g. ["TOMORROW", "MOM", "STORE", "GO"] → "Tomorrow Mom will
go to the store." The English is kept for a later gloss → English stage; the
recogniser trains on the glosses only.

Word order follows common ASL patterns: time first, topic before comment,
wh-signs last, negation before the verb or adjective. The templates were
written by a non-signer and have not been reviewed by a fluent signer — treat
them as plausible orderings, not as ASL grammar. Non-manual markers (question
faces etc.) cannot be produced by stitching citation-form clips at all.

The vocabulary has no YOU, I/ME, HOW, WANT or NAME, so many everyday phrases
cannot be formed. A fraction of samples are random sign sequences instead of
templates (see PhraseGrammar.sample): they cover every sign and keep the
recogniser from leaning on the templates' word order.

Sign names are the keys of sign_to_prediction_index_map.json (lower case).

Run as a script to print samples and vocabulary coverage:
  PYTHONPATH=research/models uv run python -m cnn_transformer.data.phrase_grammar
"""

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

FAMILY = ["mom", "dad", "grandma", "grandpa", "brother", "aunt", "uncle"]
PEOPLE = ["boy", "girl", "child", "man", "person", "fireman", "police", "clown", "cowboy"]
ANIMALS = ["alligator", "bee", "bird", "cat", "cow", "dog", "donkey", "duck", "elephant", "fish",
           "frog", "giraffe", "goose", "hen", "horse", "kitty", "lion", "mouse", "owl", "pig",
           "puppy", "tiger", "wolf", "zebra", "bug"]
FOODS = ["apple", "carrot", "cereal", "chocolate", "frenchfries", "gum", "icecream", "milk",
         "nuts", "pizza", "snack", "water", "food"]
DRINKS = ["milk", "water"]
TOYS = ["balloon", "book", "doll", "toy", "puzzle", "gift", "flag", "flower"]
CLOTHES = ["hat", "jacket", "jeans", "mitten", "shirt", "shoe", "pajamas", "underwear", "zipper"]
THINGS = ["pen", "pencil", "penny", "napkin", "scissors", "toothbrush", "lamp", "radio", "TV",
          "chair", "table", "bed", "drawer", "closet", "refrigerator", "vacuum", "garbage",
          "glasswindow", "boat", "car", "airplane", "helicopter", "dryer", "stairs", "tree"]
VEHICLES = ["boat", "car", "airplane", "helicopter", "horse"]
CONTAINERS = ["drawer", "closet", "refrigerator", "garbage", "pool"]
PLACES = ["backyard", "bedroom", "farm", "home", "outside", "pool", "room", "store"]
NATURE = ["tree", "grass", "flower", "moon", "sun", "cloud"]
BODY = ["arm", "cheek", "chin", "ear", "eye", "face", "feet", "finger", "hair", "head", "lips",
        "mouth", "nose", "tongue", "tooth"]
FEELINGS = ["happy", "sad", "mad", "sick", "sleepy", "hungry", "thirsty", "awake", "fine",
            "better", "cute", "pretty", "old", "quiet", "noisy", "loud", "fast"]
THING_ADJ = ["hot", "wet", "dry", "dirty", "clean", "sticky", "yucky", "empty", "pretty", "old",
             "cute", "bad", "loud", "high", "stuck"]
COLORS = ["black", "blue", "brown", "green", "orange", "red", "white", "yellow"]
TIMES = ["now", "later", "tomorrow", "yesterday", "morning", "night"]
ACTIONS = ["cry", "dance", "jump", "sleep", "nap", "wait", "stay", "hide", "smile", "talk",
           "read", "fall", "ride", "wake", "pretend", "think", "callonphone"]
ROUTINES = ["bath", "shower", "nap", "snack", "story", "potty", "sleep"]
WEATHER = ["rain", "snow", "sun", "cloud"]
LIKE_VERBS = ["like", "hate"]
PERCEIVE = ["see", "look", "hear", "listen", "find", "have", "make", "give", "touch", "taste",
            "kiss", "drop", "cut", "open", "close", "clean", "dry", "blow"]
GREET = ["hello", "bye", "thankyou"]

# English surface forms (default: the gloss itself).
_EN = {
    "frenchfries": "French fries", "icecream": "ice cream", "glasswindow": "window",
    "callonphone": "call on the phone", "thankyou": "thank you", "haveto": "have to",
    "TV": "TV", "feet": "feet", "underwear": "underwear", "pajamas": "pajamas", "jeans": "jeans",
    "scissors": "scissors", "nuts": "nuts", "potty": "the potty", "story": "a story",
    "bath": "a bath", "shower": "a shower", "nap": "a nap", "snack": "a snack", "sleep": "sleep",
}
_PLURAL_NOUNS = {"jeans", "pajamas", "scissors", "nuts", "feet", "frenchfries", "underwear", "stairs"}
_PLURALS = {"mouse": "mice", "fish": "fish", "goose": "geese", "wolf": "wolves", "puppy": "puppies",
            "kitty": "kitties", "child": "children", "man": "men", "person": "people",
            "fireman": "firemen", "police": "police", "sheep": "sheep", "potty": "potties"}
_MASS_NOUNS = {"milk", "water", "food", "gum", "cereal", "chocolate", "icecream", "pizza", "hair"}
_THIRD = {"have": "has", "go": "goes", "cry": "cries", "kiss": "kisses", "touch": "touches",
          "watch": "watches", "fly": "flies"}
_ING = {"cry": "crying", "dance": "dancing", "jump": "jumping", "sleep": "sleeping",
        "nap": "napping", "wait": "waiting", "stay": "staying", "hide": "hiding",
        "smile": "smiling", "talk": "talking", "read": "reading", "fall": "falling",
        "ride": "riding", "wake": "waking up", "pretend": "pretending", "think": "thinking",
        "callonphone": "calling on the phone"}
_PAST = {"cry": "cried", "dance": "danced", "jump": "jumped", "sleep": "slept", "nap": "napped",
         "wait": "waited", "stay": "stayed", "hide": "hid", "smile": "smiled", "talk": "talked",
         "read": "read", "fall": "fell", "ride": "rode", "wake": "woke up", "pretend": "pretended",
         "think": "thought", "callonphone": "called on the phone", "rain": "rained",
         "snow": "snowed"}


def en(g: str) -> str:
    return _EN.get(g, g)


def _noun(g: str, poss: bool = False) -> Tuple[str, bool]:
    """English noun phrase and whether it takes plural agreement."""
    plural = g in _PLURAL_NOUNS
    if poss:
        return f"my {en(g)}", plural
    if g in FAMILY:
        return en(g).capitalize(), False
    if plural or g in _MASS_NOUNS:
        return f"the {en(g)}", plural
    return f"the {en(g)}", False


def _plural(g: str) -> str:
    """Generic English plural of a gloss ("I like ___")."""
    if g in _MASS_NOUNS or g in _PLURAL_NOUNS or g in WEATHER:
        return en(g)
    if g in _PLURALS:
        return _PLURALS[g]
    w = en(g)
    return w + ("es" if w.endswith(("s", "sh", "ch", "x")) else "s")


def _third(v: str) -> str:
    if v in _THIRD:
        return _THIRD[v]
    return v + ("es" if v.endswith(("s", "sh", "ch", "x")) else "s")


def _sentence(s: str, q: bool = False) -> str:
    s = s.strip()
    return s[0].upper() + s[1:] + ("?" if q else ".")


Sample = Tuple[List[str], str]


class PhraseGrammar:
    """Samples (glosses, english) pairs restricted to an available sign set.

    available: the signs one signer has clips for (phrases use one signer, so a
    template slot only draws from that signer's signs). Templates whose slots
    cannot be filled are skipped; if none can, sample() falls back to a random
    sequence.
    """

    def __init__(self, vocab: Sequence[str]):
        self.vocab = list(vocab)
        unknown = {w for lst in _ALL_LISTS for w in lst} - set(self.vocab)
        assert not unknown, f"grammar words missing from the sign map: {sorted(unknown)}"
        self.templates: List[Callable] = [
            self._t_describe, self._t_thing_adj, self._t_color, self._t_time_go,
            self._t_where, self._t_who, self._t_why, self._t_like, self._t_not_like,
            self._t_topic_hate, self._t_request, self._t_routine, self._t_weather,
            self._t_greet, self._t_finish, self._t_my_where, self._t_perceive,
            self._t_body_owie, self._t_into, self._t_on_table, self._t_haveto, self._t_can,
            self._t_answer, self._t_ride, self._t_drink, self._t_shhh, self._t_say,
            self._t_that_mine, self._t_many, self._t_nature, self._t_up_down,
        ]

    # ---- slot helpers -------------------------------------------------------
    def _pick(self, rng, words, avail: Set[str], exclude=()):
        opts = [w for w in words if w in avail and w not in exclude]
        if not opts:
            raise _NoFill
        return opts[rng.integers(len(opts))]

    def _subject(self, rng, avail, allow_pron=True, allow_poss=True, animals=True):
        """(glosses, english, plural) for a person (or animal) subject."""
        r = rng.random()
        if allow_pron and r < 0.25 and "hesheit" in avail:
            return ["hesheit"], ["he", "she"][rng.integers(2)], False
        if allow_pron and r < 0.35 and "weus" in avail:
            return ["weus"], "we", True
        g = self._pick(rng, FAMILY + PEOPLE + (ANIMALS if animals else []), avail)
        poss = allow_poss and g not in FAMILY and "minemy" in avail and rng.random() < 0.3
        text, plural = _noun(g, poss)
        return (["minemy", g] if poss else [g]), text, plural

    # ---- templates (each returns (glosses, english)) -------------------------
    def _t_describe(self, rng, a):  # MINEMY DOG HUNGRY
        s, t, pl = self._subject(rng, a)
        adj = self._pick(rng, FEELINGS, a)
        return s + [adj], _sentence(f"{t} {'are' if pl else 'is'} {adj}")

    def _t_thing_adj(self, rng, a):  # SHIRT DIRTY
        n = self._pick(rng, TOYS + CLOTHES + THINGS + FOODS, a)
        adj = self._pick(rng, THING_ADJ, a)
        poss = "minemy" in a and rng.random() < 0.3
        t, pl = _noun(n, poss)
        return (["minemy"] if poss else []) + [n, adj], _sentence(f"{t} {'are' if pl else 'is'} {en(adj)}")

    def _t_color(self, rng, a):  # CAR RED
        n = self._pick(rng, TOYS + CLOTHES + VEHICLES + ANIMALS + ["flower", "apple", "carrot"], a)
        c = self._pick(rng, COLORS, a)
        t, pl = _noun(n)
        return [n, c], _sentence(f"{t} {'are' if pl else 'is'} {c}")

    def _t_time_go(self, rng, a):  # TOMORROW MOM STORE GO
        tm = self._pick(rng, ["tomorrow", "later", "yesterday", "now", "morning"], a)
        s, t, pl = self._subject(rng, a, allow_poss=False)
        p = self._pick(rng, PLACES, a, exclude=("room",))
        self._pick(rng, ["go"], a)
        dest = {"home": "home", "outside": "outside"}.get(p, f"to the {en(p)}")
        verb = {"yesterday": "went", "now": "is going" if not pl else "are going",
                "morning": "will go"}.get(tm, "will go")
        when = {"morning": "in the morning"}.get(tm, tm)
        return [tm] + s + [p, "go"], _sentence(f"{when} {t} {verb} {dest}" if tm != "now"
                                               else f"{t} {verb} {dest} now")

    def _t_where(self, rng, a):  # DOG WHERE
        n = self._pick(rng, FAMILY + ANIMALS + TOYS + CLOTHES + THINGS, a)
        t, pl = _noun(n)
        self._pick(rng, ["where"], a)
        return [n, "where"], _sentence(f"where {'are' if pl else 'is'} {t}", q=True)

    def _t_who(self, rng, a):  # CRY WHO
        v = self._pick(rng, ACTIONS, a, exclude=("read",))
        self._pick(rng, ["who"], a)
        return [v, "who"], _sentence(f"who is {_ING[v]}", q=True)

    def _t_why(self, rng, a):  # MOM SAD WHY
        s, t, pl = self._subject(rng, a, allow_poss=False, animals=False)
        adj = self._pick(rng, ["sad", "mad", "happy", "sick", "sleepy", "hungry", "thirsty", "quiet", "noisy"], a)
        self._pick(rng, ["why"], a)
        return s + [adj, "why"], _sentence(f"why {'are' if pl else 'is'} {t} {adj}", q=True)

    def _t_like(self, rng, a):  # HESHEIT LIKE PIZZA
        s, t, pl = self._subject(rng, a, animals=False)
        v = self._pick(rng, LIKE_VERBS, a)
        o = self._pick(rng, FOODS + ANIMALS + TOYS + WEATHER, a, exclude=s)
        ot = _plural(o)
        return s + [v, o], _sentence(f"{t} {v if pl else _third(v)} {'the ' + ot if o in WEATHER else ot}")

    def _t_not_like(self, rng, a):  # WEUS NOT LIKE CARROT
        s, t, pl = self._subject(rng, a, animals=False)
        self._pick(rng, ["not"], a)
        self._pick(rng, ["like"], a)
        o = self._pick(rng, FOODS + ANIMALS + TOYS, a, exclude=s)
        ot = _plural(o)
        return s + ["not", "like", o], _sentence(f"{t} {'do' if pl else 'does'} not like {ot}")

    def _t_topic_hate(self, rng, a):  # CARROT, HESHEIT HATE
        o = self._pick(rng, FOODS + ANIMALS + ["bath", "shower", "rain", "snow", "vacuum"], a)
        s, t, pl = self._subject(rng, a, allow_poss=False, animals=False)
        self._pick(rng, ["hate"], a)
        ot = {"bath": "baths", "shower": "showers", "rain": "the rain", "snow": "snow",
              "vacuum": "the vacuum"}.get(o, _plural(o))
        return [o] + s + ["hate"], _sentence(f"{t} {'hate' if pl else 'hates'} {ot}")

    def _t_request(self, rng, a):  # MILK PLEASE / PLEASE GIVE BOOK
        self._pick(rng, ["please"], a)
        if "give" in a and rng.random() < 0.5:
            o = self._pick(rng, FOODS + TOYS + THINGS[:6], a)
            return ["please", "give", o], _sentence(f"please give me {_noun(o)[0]}")
        o = self._pick(rng, FOODS + TOYS, a)
        return [o, "please"], _sentence(f"{en(o)}, please")

    def _t_routine(self, rng, a):  # NOW BATH / LATER STORY
        tm = self._pick(rng, ["now", "later", "night", "morning"], a)
        r = self._pick(rng, ROUTINES, a)
        when = {"night": "tonight", "morning": "in the morning"}.get(tm, tm)
        return [tm, r], _sentence(f"time for {en(r)} {when}" if r != "sleep" else f"time to sleep {when}")

    def _t_weather(self, rng, a):  # YESTERDAY SNOW / OUTSIDE HOT
        if "outside" in a and rng.random() < 0.3:
            adj = self._pick(rng, ["hot", "wet", "dirty", "noisy", "quiet"], a)
            return ["outside", adj], _sentence(f"it is {adj} outside")
        tm = self._pick(rng, ["tomorrow", "yesterday", "now", "morning", "night"], a)
        w = self._pick(rng, WEATHER, a)
        if w in ("sun", "cloud"):
            return [tm, w], _sentence({"sun": "it is sunny", "cloud": "it is cloudy"}[w] + f" {tm}")
        verb = {"yesterday": _PAST[w], "tomorrow": f"will {w}"}.get(tm, f"is {w}ing")
        when = {"morning": "this morning", "night": "tonight"}.get(tm, tm)
        return [tm, w], _sentence(f"it {verb} {when}")

    def _t_greet(self, rng, a):  # HELLO GRANDMA
        g = self._pick(rng, GREET, a)
        n = self._pick(rng, FAMILY, a)
        return [g, n], _sentence(f"{en(g)}, {en(n).capitalize()}")

    def _t_finish(self, rng, a):  # HESHEIT SLEEP FINISH
        s, t, pl = self._subject(rng, a, allow_poss=False, animals=False)
        v = self._pick(rng, ["sleep", "nap", "read", "dance", "talk", "callonphone", "cry"], a)
        self._pick(rng, ["finish"], a)
        return s + [v, "finish"], _sentence(f"{t} {'have' if pl else 'has'} finished {_ING[v]}")

    def _t_my_where(self, rng, a):  # MINEMY SHOE WHERE
        self._pick(rng, ["minemy"], a)
        self._pick(rng, ["where"], a)
        n = self._pick(rng, TOYS + CLOTHES + THINGS + ["dog", "cat", "kitty", "puppy"], a)
        pl = n in _PLURAL_NOUNS
        return ["minemy", n, "where"], _sentence(f"where {'are' if pl else 'is'} my {en(n)}", q=True)

    def _t_perceive(self, rng, a):  # WEUS SEE ELEPHANT
        s, t, pl = self._subject(rng, a, animals=False)
        v = self._pick(rng, PERCEIVE, a)
        pool = {"hear": ANIMALS + ["radio", "TV"], "listen": ["radio", "TV"] + FAMILY,
                "taste": FOODS, "blow": ["balloon"], "cut": ["apple", "carrot", "hair", "pizza"],
                "open": CONTAINERS + ["gift", "book", "glasswindow"],
                "close": CONTAINERS[:3] + ["book", "glasswindow"],
                "clean": ["room", "bedroom", "table", "car", "chair"],
                "dry": ["hair", "shirt", "jeans", "feet"], "kiss": FAMILY + ["dog", "cat", "kitty", "puppy"],
                "make": FOODS[:10] + ["puzzle", "flag"]}.get(v, ANIMALS + TOYS + THINGS + NATURE + ["animal"])
        o = self._pick(rng, pool, a, exclude=s)
        vt = {"look": "look at", "listen": "listen to"}.get(v, v)
        if not pl:
            vt = " ".join([_third(vt.split()[0])] + vt.split()[1:])
        return s + [v, o], _sentence(f"{t} {vt} {_noun(o)[0]}")

    def _t_body_owie(self, rng, a):  # MINEMY ARM OWIE
        self._pick(rng, ["owie"], a)
        b = self._pick(rng, BODY, a)
        poss = "minemy" in a and rng.random() < 0.7
        t, pl = _noun(b, poss)
        return (["minemy"] if poss else []) + [b, "owie"], _sentence(f"{t} {'hurt' if pl else 'hurts'}")

    def _t_into(self, rng, a):  # MILK INTO REFRIGERATOR
        self._pick(rng, ["into"], a)
        o = self._pick(rng, FOODS + TOYS + CLOTHES, a)
        c = self._pick(rng, CONTAINERS, a)
        return [o, "into", c], _sentence(f"put {_noun(o)[0]} into the {en(c)}")

    def _t_on_table(self, rng, a):  # BOOK ON TABLE
        self._pick(rng, ["on"], a)
        o = self._pick(rng, TOYS + FOODS + THINGS[:8], a, exclude=("table",))
        p = self._pick(rng, ["table", "chair", "bed"], a)
        t, pl = _noun(o)
        return [o, "on", p], _sentence(f"{t} {'are' if pl else 'is'} on the {p}")

    def _t_haveto(self, rng, a):  # WEUS HAVETO WAIT
        s, t, pl = self._subject(rng, a, allow_poss=False, animals=False)
        self._pick(rng, ["haveto"], a)
        v = self._pick(rng, ["wait", "sleep", "go", "stay", "clean", "read", "talk", "nap"], a)
        return s + ["haveto", v], _sentence(f"{t} {'have' if pl else 'has'} to {v}")

    def _t_can(self, rng, a):  # GIRL CAN JUMP / HESHEIT WILL DANCE
        s, t, pl = self._subject(rng, a, allow_poss=False, animals=False)
        m = self._pick(rng, ["can", "will"], a)
        v = self._pick(rng, ["jump", "dance", "read", "ride", "hide", "talk", "wait", "stay"], a)
        neg = "not" in a and rng.random() < 0.3
        return s + [m] + (["not"] if neg else []) + [v], _sentence(
            f"{t} {m}{'not' if neg and m == 'can' else ' not' if neg else ''} {v}")

    def _t_answer(self, rng, a):  # YES, HUNGRY / NO, THIRSTY NOT
        yn = self._pick(rng, ["yes", "no"], a)
        adj = self._pick(rng, ["hungry", "thirsty", "sleepy", "sick", "fine", "happy", "sad"], a)
        if yn == "no" and "not" in a:
            return ["no", adj, "not"], _sentence(f"no, I am not {adj}")
        return [yn, adj], _sentence(f"{yn}, I am {adj}")

    def _t_ride(self, rng, a):  # TOMORROW AIRPLANE RIDE
        tm = self._pick(rng, ["tomorrow", "yesterday", "later", "now"], a)
        v = self._pick(rng, ["boat", "car", "airplane", "helicopter", "horse"], a)
        self._pick(rng, ["ride"], a)
        verb = {"yesterday": "rode", "now": "are riding"}.get(tm, "will ride")
        return [tm, v, "ride"], _sentence(f"we {verb} in {_noun(v)[0]} {tm}" if v != "horse"
                                          else f"we {verb} a horse {tm}")

    def _t_drink(self, rng, a):  # THIRSTY, WATER DRINK
        self._pick(rng, ["thirsty"], a)
        self._pick(rng, ["drink"], a)
        d = self._pick(rng, DRINKS, a)
        return ["thirsty", d, "drink"], _sentence(f"I am thirsty, I will drink {en(d)}")

    def _t_shhh(self, rng, a):  # SHHH CHILD SLEEP
        self._pick(rng, ["shhh"], a)
        s, t, pl = self._subject(rng, a, allow_pron=False, allow_poss=False)
        v = self._pick(rng, ["sleep", "nap", "read", "listen"], a)
        return ["shhh"] + s + [v], _sentence(f"shh, {t} {'are' if pl else 'is'} {_ING.get(v, v + 'ing')}")

    def _t_say(self, rng, a):  # MOM SAY NO
        s, t, pl = self._subject(rng, a, allow_poss=False, animals=False)
        self._pick(rng, ["say"], a)
        yn = self._pick(rng, ["yes", "no"], a)
        return s + ["say", yn], _sentence(f"{t} {'say' if pl else 'says'} {yn}")

    def _t_that_mine(self, rng, a):  # THAT MINEMY BOOK
        self._pick(rng, ["that"], a)
        self._pick(rng, ["minemy"], a)
        n = self._pick(rng, TOYS + CLOTHES + THINGS[:10] + ["dog", "cat", "kitty", "puppy"], a)
        return ["that", "minemy", n], _sentence(f"{'those are' if n in _PLURAL_NOUNS else 'that is'} my {en(n)}")

    def _t_many(self, rng, a):  # FARM ANIMAL MANY
        p = self._pick(rng, ["farm", "store", "backyard", "outside"], a)
        n = self._pick(rng, ["animal", "cow", "pig", "horse", "hen", "toy", "book", "flower", "tree", "bug", "bird"], a)
        self._pick(rng, ["many"], a)
        where = {"outside": "outside", "farm": "on the farm"}.get(p, f"in the {p}")
        return [p, n, "many"], _sentence(f"there are many {_plural(n)} {where}")

    def _t_nature(self, rng, a):  # NIGHT MOON / GRASS GREEN / THERE TREE
        if "there" in a and rng.random() < 0.4:
            n = self._pick(rng, NATURE + ANIMALS, a)
            return ["there", n], _sentence(f"there is {'the' if n in ('moon', 'sun') else 'a'} {en(n)}")
        n = self._pick(rng, NATURE, a)
        c = self._pick(rng, COLORS, a)
        return [n, c], _sentence(f"the {en(n)} is {c}")

    def _t_up_down(self, rng, a):  # STAIRS UP GO / BALLOON UP
        d = self._pick(rng, ["up", "down"], a)
        if "stairs" in a and "go" in a and rng.random() < 0.5:
            s, t, pl = self._subject(rng, a, allow_poss=False, animals=False)
            return s + ["stairs", d, "go"], _sentence(f"{t} {'go' if pl else 'goes'} {d} the stairs")
        n = self._pick(rng, ["balloon", "airplane", "helicopter", "bird", "sun", "moon"], a)
        return [n, d], _sentence(f"the {en(n)} {'goes' } {d}")

    # ---- public ---------------------------------------------------------------
    def sample(self, rng: np.random.Generator, available: Optional[Set[str]] = None,
               random_frac: float = 0.3, max_len: int = 5) -> Sample:
        """One (glosses, english) pair using only `available` signs (default: all).

        With probability random_frac — or when no template can be filled — the
        phrase is 1–max_len signs drawn uniformly from `available` with no
        immediate repeats, and its English is the glosses joined by spaces.
        """
        avail = set(self.vocab) if available is None else set(available)
        if rng.random() >= random_frac:
            for k in rng.permutation(len(self.templates)):
                try:
                    g, e = self.templates[k](rng, avail)
                except _NoFill:
                    continue
                if len(g) <= max_len and all(x != y for x, y in zip(g, g[1:])):
                    return g, e
        words = sorted(avail)
        n = int(rng.integers(1, max_len + 1)) if len(words) > 1 else 1  # no immediate repeats
        g: List[str] = []
        while len(g) < n:
            w = words[rng.integers(len(words))]
            if not g or w != g[-1]:
                g.append(w)
        return g, " ".join(x.upper() for x in g)


class _NoFill(Exception):
    pass


_ALL_LISTS = [FAMILY, PEOPLE, ANIMALS, FOODS, DRINKS, TOYS, CLOTHES, THINGS, VEHICLES, CONTAINERS,
              PLACES, BODY, FEELINGS, THING_ADJ, COLORS, TIMES, ACTIONS, ROUTINES, WEATHER,
              LIKE_VERBS, PERCEIVE, GREET,
              NATURE, ["shhh", "say", "that", "many", "animal", "there", "up", "down", "stairs"],
              ["go", "where", "who", "why", "not", "please", "give", "finish", "minemy", "hesheit",
               "weus", "owie", "into", "on", "haveto", "can", "will", "yes", "no", "drink", "ride"]]


def _main():
    import json
    from collections import Counter
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    for cand in ("data/asl-is-lmdb", "data/Isolated_ASL_Recognition"):
        p = root / cand / "sign_to_prediction_index_map.json"
        if p.exists():
            vocab = list(json.load(open(p)))
            break
    else:
        raise SystemExit("sign_to_prediction_index_map.json not found")
    gr = PhraseGrammar(vocab)
    rng = np.random.default_rng(0)
    for _ in range(25):
        g, e = gr.sample(rng, random_frac=0.0)
        print(f"  {' '.join(x.upper() for x in g):<40} {e}")
    in_templates, lens = Counter(), Counter()
    for _ in range(20000):
        g, _ = gr.sample(rng, random_frac=0.0)
        in_templates.update(g)
        lens[len(g)] += 1
    never = sorted(set(vocab) - set(in_templates))
    print(f"\nTemplate phrases: lengths {dict(sorted(lens.items()))}; "
          f"{len(set(in_templates))}/{len(vocab)} signs used; never in a template "
          f"({len(never)}, covered only by random sequences): {', '.join(never)}")


if __name__ == "__main__":
    _main()

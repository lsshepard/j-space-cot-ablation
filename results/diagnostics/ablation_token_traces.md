# Ablation token-trace qualitative report

band=[27,33] k=10 exclude_topk=10 n=10

## Cross-problem summary
```json
{
  "n": 10,
  "mean_survivor_fraction": 0.9970170030242835,
  "mean_n_excluded": 0.02982996975716413,
  "mean_delta_h_norm": 70.64506531510817,
  "problems_with_intermediate_survivor": 10,
  "problems_with_intermediate_excluded": 5,
  "problems_with_intermediate_in_clean_topk": 10,
  "problems_where_completion_changed": 7,
  "per_problem": [
    {
      "problem_id": "super-populous-capital",
      "gold": "Beijing",
      "intermediates": [
        "China"
      ],
      "clean_completion": "Beijing.",
      "jspace_completion": "Paris is the capital city of the most populous country in the world, which is",
      "mean_survivor_fraction": 0.9975683890577507,
      "mean_n_excluded": 0.0243161094224924,
      "mean_delta_h_norm": 70.64815068462337,
      "n_intermediate_survivor_hits": 297,
      "n_intermediate_excluded_hits": 7,
      "n_intermediate_clean_topk_hits": 50,
      "last_pos_survivor_tokens": [
        " enam",
        " n\u00e2",
        "\u2026\\n",
        "\u3002\\",
        " \uff0c",
        "\u6c5f\u82cf\u7701",
        "\u00a0",
        " *\"",
        "\uff0c",
        "\u2026\\n\\n\\n",
        "-*",
        ".\";\\n",
        "\u00ab",
        "cycl",
        "\uff1a\\n",
        " [",
        "\u201d\u2014",
        "\u2026\\n\\n",
        ".(",
        " Hawai",
        " enam",
        "...\\n\\n\\n",
        "\":\\n",
        "?\"",
        "\u5fb7\u56fd",
        " $",
        "\u4f9d",
        "\u201d;",
        "**,",
        " Ukraine",
        "\":\\r\\n",
        " ...)",
        " {\"",
        "\">#",
        " _",
        "{\\n",
        "...\"\\n",
        "\u0e32\u0e19",
        "`,\\n",
        " ...\"\\n\\n",
        " ...\"\\n\\n",
        " PPP",
        "\u68f5\u6811",
        "'\\n\\n\\n",
        "):\\r\\n",
        " controls",
        " ...\\n\\n",
        " Blogger",
        " \u00a0 \u00a0 \u00a0 \u00a0",
        "logfile",
        " enam",
        "\ufffd",
        "\u90ae",
        "\u4e0d\u660e",
        " [\u2026]",
        " Int",
        " \\",
        "atee",
        "\u6bd7",
        "Ti",
        " enam",
        " \ufffd",
        "\uff0c",
        "\u2026\\n\\n\\n",
        ",\\n\\n\\n",
        " sy",
        " ton",
        " _(",
        "')[",
        "\u4ea4\u6362"
      ],
      "last_pos_excluded_tokens": [],
      "last_pos_clean_topk": [
        "Be",
        "Berlin",
        "China",
        "London",
        "New",
        "Paris",
        "Tok",
        "Wars",
        "Washington",
        "\u5317\u4eac"
      ]
    },
    {
      "problem_id": "amazon-language",
      "gold": "Portuguese",
      "intermediates": [
        "Brazil"
      ],
      "clean_completion": "Spanish.",
      "jspace_completion": "Spanish.",
      "mean_survivor_fraction": 0.9969604863221885,
      "mean_n_excluded": 0.030395136778115502,
      "mean_delta_h_norm": 71.07755236350295,
      "n_intermediate_survivor_hits": 9,
      "n_intermediate_excluded_hits": 1,
      "n_intermediate_clean_topk_hits": 11,
      "last_pos_survivor_tokens": [
        " enam",
        "\u3002\\",
        " n\u00e2",
        "\u2026\\n",
        " \uff0c",
        "...\\n\\n",
        "...\"\\n",
        "\uff0c",
        "\u533b\u5b66",
        "\u6c5f\u82cf\u7701",
        " ()\\n\\n",
        "\u300d\\n\\n",
        ";\\n\\n\\n\\n",
        " enam",
        " Jean",
        "\u822a\u6d77",
        "...\"\\n",
        " `'",
        "\u00a0",
        "\uff1b",
        "\u00a0",
        " ...\"\\n\\n",
        " Brazilian",
        "\u773c\u754c",
        "?\"",
        "\":\\n",
        "\u00a0\\n",
        "construction",
        "...\\n\\n\\n",
        " \"\"\"\\n\\n",
        "]\\n\\n\\n",
        " ...)",
        " \"{{",
        "}@",
        ",...\\n\\n",
        " penet",
        "<|endoftext|>",
        " \\",
        "\u2013",
        " **",
        "\"\\n\\n",
        " subjects",
        "';",
        "\u68f5\u6811",
        " #",
        "\u5982\u4f55\u770b\u5f85",
        "publication",
        "\u7ae3",
        "\u5757",
        " Savannah",
        " ...\"\\n\\n",
        ".\"[",
        " \\",
        " enam",
        " {$",
        "\ufffd",
        " \\",
        " \\n",
        " [[",
        " proxies",
        " \ufffd",
        " **",
        " *",
        "=wx",
        " FB",
        "\u2026\\n\\n\\n",
        "    ",
        " (`",
        "\u63a5\u53d7",
        "asive"
      ],
      "last_pos_excluded_tokens": [],
      "last_pos_clean_topk": [
        " Portuguese",
        " Spanish",
        "Brazil",
        "Esp",
        "French",
        "Port",
        "Span",
        "Spanish",
        "\u8461\u8404\u7259",
        "\u897f\u73ed\u7259"
      ]
    },
    {
      "problem_id": "super-smallest-continent",
      "gold": "Europe",
      "intermediates": [
        "Vatican"
      ],
      "clean_completion": "Europe.",
      "jspace_completion": "Europe",
      "mean_survivor_fraction": 0.9969604863221885,
      "mean_n_excluded": 0.030395136778115502,
      "mean_delta_h_norm": 71.68366531325691,
      "n_intermediate_survivor_hits": 290,
      "n_intermediate_excluded_hits": 8,
      "n_intermediate_clean_topk_hits": 43,
      "last_pos_survivor_tokens": [
        " enam",
        " n\u00e2",
        "\u3002\\",
        "\u2026\\n",
        " \uff0c",
        "\u00a0",
        " *\"",
        "...\"\\n",
        "\u00a0",
        ":\\n",
        ";</",
        "\u00ab",
        " \ufffd",
        " enam",
        " (<",
        "**\\n\\n",
        " park",
        "\u535a\u7269\u9986",
        " physicians",
        "asury",
        "\u661f\u7403",
        "conditions",
        " ()=>",
        "**,",
        "\u83b2\u82b1",
        " \\n",
        "\u7684\u90e8\u5206",
        "\u3002\\",
        "!</",
        " Fib",
        " enam",
        " nest",
        ":\\n\\n\\n",
        " :",
        "*/,\\n",
        " __",
        "pane",
        " buffer",
        "?</",
        "`\\n\\n",
        "1",
        ",",
        " eating",
        " clich",
        "\u5229\u6da6\u7387",
        "\u304f\u308a",
        ".\u201d\\n\\n\\n\\n",
        "\u2026\\n\\n\\n\\n",
        " merging",
        "obar",
        " enam",
        ".oper",
        " [[",
        " oper",
        " *",
        ".Obj",
        "\u62a5",
        "aden",
        "\u5341\u5b57",
        " stack",
        "\uff1b",
        "\u2026\\n\\n\\n",
        "ZR",
        "\uff0c",
        "attr",
        " \ufffd",
        "\u3002\\",
        "*)\\n",
        " inversion",
        "\u0203"
      ],
      "last_pos_excluded_tokens": [],
      "last_pos_clean_topk": [
        "Africa",
        "Ant",
        "Asia",
        "Atlantic",
        "E",
        "EU",
        "Eu",
        "Euro",
        "Europe",
        "European"
      ]
    },
    {
      "problem_id": "atomic-26-symbol",
      "gold": "Fe",
      "intermediates": [
        "iron"
      ],
      "clean_completion": "Fe",
      "jspace_completion": "Fe",
      "mean_survivor_fraction": 0.9967261904761904,
      "mean_n_excluded": 0.03273809523809524,
      "mean_delta_h_norm": 71.06531753710338,
      "n_intermediate_survivor_hits": 309,
      "n_intermediate_excluded_hits": 7,
      "n_intermediate_clean_topk_hits": 41,
      "last_pos_survivor_tokens": [
        " enam",
        " n\u00e2",
        "\u3002\\",
        "\u2026\\n",
        " \uff0c",
        " **",
        " listeners",
        "-\\r\\n",
        "...\"\\n",
        "\u00bb.",
        "\u201d\\n",
        " **",
        " park",
        ",\\n\\n\\n\\n",
        " Gujarat",
        " ...\"\\n\\n",
        "{})",
        "plash",
        "\uff0c\u201c",
        ";\\n\\n\\n\\n",
        "\u5fb7\u56fd",
        " permission",
        "\u5e38\u5fb7",
        "\u00a0",
        "\u653f\u6cbb",
        "pers",
        " Busty",
        " \u00e2",
        "**,",
        "\u30fc\u30eb",
        " **",
        "\u6e90",
        "\u2014\"",
        "\u7f51\u7ad9\u5efa\u8bbe",
        " \u2014\\n\\n",
        " \u2026\\n\\n",
        " buffer",
        "quam",
        "').",
        "immel",
        " enam",
        " picks",
        "\ufffd",
        " pav",
        "cken",
        " gradients",
        "este",
        "lev",
        "~,",
        "isers",
        "\ufffd",
        " contracting",
        " replication",
        " Ness",
        " compuls",
        " conten",
        " {$",
        "\u4ed6\u4eec\u7684",
        " \u201c",
        "\u2026\u201d\\n\\n",
        "\u2026\\n\\n\\n",
        "\uff0c",
        " _(",
        " \uff0c",
        " lact",
        "compose",
        "\u950c",
        "\u4f2a\u88c5",
        " cath",
        " enam"
      ],
      "last_pos_excluded_tokens": [],
      "last_pos_clean_topk": [
        " Fe",
        "Co",
        "Cr",
        "FE",
        "Fe",
        "Iron",
        "Ni",
        "Nick",
        "iron",
        "\u94c1"
      ]
    },
    {
      "problem_id": "colosseum-currency",
      "gold": "Euro",
      "intermediates": [
        "Italy"
      ],
      "clean_completion": "Euro.",
      "jspace_completion": "Euro",
      "mean_survivor_fraction": 0.9974285714285714,
      "mean_n_excluded": 0.025714285714285714,
      "mean_delta_h_norm": 68.93721882138934,
      "n_intermediate_survivor_hits": 3,
      "n_intermediate_excluded_hits": 0,
      "n_intermediate_clean_topk_hits": 8,
      "last_pos_survivor_tokens": [
        " enam",
        " n\u00e2",
        " \uff0c",
        "\u3002\\",
        "\u2026\\n",
        "...\"\\n",
        "?,\\n",
        ":`",
        "\u2019.",
        "\u00a0",
        ".\";\\n",
        " enam",
        " Hawai",
        "**",
        "\u4ed6\u4eec\u7684",
        " divide",
        "!)",
        "\u98de\u884c",
        " physicians",
        " ()\\n\\n",
        " enam",
        "\uff01\\n\\n",
        "\u2014\u2014",
        "\u2019.",
        "\u83b2\u82b1",
        "\u4eba\u6027",
        ":*",
        "\u7c7b\u4f3c\u7684",
        "_TRI",
        " ineligible",
        " enam",
        "...\\n\\n",
        " \u2014\\n\\n",
        "\">#",
        "]\\n\\n\\n",
        "\u00ad",
        "\u2019",
        " \u2026\\n\\n",
        " LZ",
        "\u2014\"",
        "?)",
        " {\\n\\n\\n\\n",
        "\u201d\\n",
        "\u4e08",
        " ()\\r\\n",
        ")\u2014",
        "\u4e2d\u56fd\u7ecf\u6d4e",
        "\u2026\\n\\n\\n\\n",
        " \\`",
        "\u4ed6\u4eec\u7684",
        " enam",
        "\ufffd",
        " Cy",
        " \\",
        " PPP",
        " \u2026",
        " ...\"\\n\\n",
        "\u533b\u751f",
        " \u00ad",
        "\u964d",
        "\uff0c",
        " [",
        "...\\n\\n",
        "ahi",
        ";&",
        "PTH",
        ",\\n\\n\\n",
        " ___",
        " \u00ad",
        "agnost"
      ],
      "last_pos_excluded_tokens": [],
      "last_pos_clean_topk": [
        " Euro",
        " euro",
        "E",
        "EU",
        "EUR",
        "Euro",
        "European",
        "Italian",
        "e",
        "\u6b27\u5143"
      ]
    },
    {
      "problem_id": "planet-3-moons",
      "gold": "1",
      "intermediates": [
        "Earth"
      ],
      "clean_completion": "1",
      "jspace_completion": "1000000000000000",
      "mean_survivor_fraction": 0.9965014577259476,
      "mean_n_excluded": 0.03498542274052478,
      "mean_delta_h_norm": 69.44597323225817,
      "n_intermediate_survivor_hits": 3,
      "n_intermediate_excluded_hits": 0,
      "n_intermediate_clean_topk_hits": 10,
      "last_pos_survivor_tokens": [
        " enam",
        " n\u00e2",
        "\u3002\\",
        "...\"\\n",
        " \uff0c",
        "\u00a0",
        "\uff0c",
        ".\\r\\n",
        " {})\\n",
        " **",
        " \\",
        " park",
        " enam",
        ";\\n\\n\\n\\n",
        " _",
        "\u6ecb\u517b",
        ",\\n\\n\\n\\n",
        " snow",
        " **",
        " Omaha",
        " enam",
        " ()=>",
        " Iranians",
        " ...\"\\n\\n",
        "\u201d\u2014",
        "`](",
        " waiting",
        " <",
        "\u5b9e\u7528",
        "\uff01",
        " **",
        "\ufffd",
        ":\\n\\n\\n",
        " enam",
        "):",
        "\u4e2a\u9879\u76ee",
        "tones",
        " \\",
        "\u6700\u540e",
        " historians",
        "iates",
        " ...\\n",
        " `{",
        ".\u201d\\n\\n\\n\\n",
        "ores",
        "?\",",
        " ()\\r\\n",
        " circ",
        "<|endoftext|>",
        " enam",
        "\u201d\u2014",
        " \u00ad",
        "\ufffd",
        " \u00ad",
        " copper",
        " \\",
        "\uff1b",
        ".\"[",
        "ali",
        "\uff0c",
        " ___",
        "\u2026\\n\\n\\n",
        " \u2014",
        " tags",
        "qi",
        ")#",
        "\u6cc4\u6f0f",
        "\u610f\u8bc6",
        " '''\\r\\n"
      ],
      "last_pos_excluded_tokens": [
        "1"
      ],
      "last_pos_clean_topk": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "9",
        "Fact",
        "The"
      ]
    },
    {
      "problem_id": "carnival-ocean",
      "gold": "Atlantic",
      "intermediates": [
        "Brazil"
      ],
      "clean_completion": "Atlantic Ocean.",
      "jspace_completion": "Atlantic Ocean",
      "mean_survivor_fraction": 0.9971988795518207,
      "mean_n_excluded": 0.028011204481792718,
      "mean_delta_h_norm": 68.48419443232005,
      "n_intermediate_survivor_hits": 5,
      "n_intermediate_excluded_hits": 0,
      "n_intermediate_clean_topk_hits": 12,
      "last_pos_survivor_tokens": [
        " enam",
        "\u3002\\",
        "\u2026\\n",
        " n\u00e2",
        " \uff0c",
        "\u533b\u5b66",
        "\u00a0",
        "\u201d\u2014",
        "\uff0c",
        "\u00a0",
        ";'\\n",
        " enam",
        "\uff1a\\n",
        " matchmaking",
        "\u822a\u6d77",
        ",\\n\\n\\n\\n",
        " \\",
        ";</",
        " \u2013",
        "\",\\n\\n",
        " enam",
        ".\"\\n",
        " Russia",
        " \u2026\\n\\n",
        "\u00a0",
        "...\\n\\n\\n",
        "?\"",
        " ());\\n",
        "square",
        "*.",
        ":\\n\\n\\n",
        " ...)",
        "\u2013",
        " ...\"\\n\\n",
        " spindle",
        "`,\\n",
        " semester",
        "\u4e5f\u53ea\u80fd",
        "')\\n\\n\\n",
        "\u2014\"",
        " #",
        "paired",
        "\u59ca\u59b9",
        " ...\"\\n\\n",
        "\u3002\\",
        "eea",
        "...'",
        " reason",
        ",",
        " tasks",
        " enam",
        "/\\n\\n\\n\\n",
        "\u89e3\u7b54",
        "idis",
        "bull",
        " declarations",
        " matlab",
        " (&",
        " shadow",
        " \\",
        "\uff0c",
        " \uff0c",
        " \ufffd",
        " \u00ad",
        "nu",
        " walker",
        "enus",
        ";&",
        "\u65e8",
        " **"
      ],
      "last_pos_excluded_tokens": [],
      "last_pos_clean_topk": [
        " Atlantic",
        " Caribbean",
        "A",
        "At",
        "Atl",
        "Atlantic",
        "Car",
        "English",
        "North",
        "Spanish"
      ]
    },
    {
      "problem_id": "atomic-29-symbol",
      "gold": "Cu",
      "intermediates": [
        "copper"
      ],
      "clean_completion": "Cu",
      "jspace_completion": "Cu",
      "mean_survivor_fraction": 0.9967261904761904,
      "mean_n_excluded": 0.03273809523809524,
      "mean_delta_h_norm": 71.12292855978012,
      "n_intermediate_survivor_hits": 5,
      "n_intermediate_excluded_hits": 0,
      "n_intermediate_clean_topk_hits": 9,
      "last_pos_survivor_tokens": [
        " enam",
        " n\u00e2",
        "\u3002\\",
        "\u2026\\n",
        " \uff0c",
        " **",
        "-\\r\\n",
        "...\"\\n",
        "\uff0c",
        " listeners",
        " park",
        " `\"",
        " **",
        "\uff0c\u201c",
        "\u4f53\u80b2\u9986",
        "\u4ed6\u4eec\u7684",
        ";\\n\\n\\n\\n",
        "\u201d\\n",
        " Gujarat",
        "\ube59",
        " ...\"\\n\\n",
        "\u5fb7\u56fd",
        "\u00a0",
        "\u2026\\n",
        ":\\n\\n\\n",
        "].\\n\\n",
        "\u5e38\u5fb7",
        " permission",
        "\u9a91\u884c",
        "\u2014\u2014",
        "\u00f3b",
        "\u2014\"",
        "\u044e\u0440",
        "-hot",
        " **",
        ":\\n\\n\\n",
        " Ridley",
        " mate",
        "\u6e90",
        "mans",
        "\u5723\u8bde",
        " enam",
        " Emit",
        " append",
        " memes",
        " picks",
        "\u2026\\n\\n",
        "riba",
        "-flex",
        " letters",
        " enam",
        " \u2026",
        " \\",
        " oper",
        "\u56fe",
        "dy",
        "\u201d\u2014",
        "actionDate",
        ".Compare",
        " setId",
        "\uff0c",
        "\u2026\\n\\n\\n",
        "\u6cc4\u6f0f",
        " \uff0c",
        " \u00ad",
        " intents",
        " lact",
        "\">{",
        "\u2026\\n\\n",
        ".CL"
      ],
      "last_pos_excluded_tokens": [],
      "last_pos_clean_topk": [
        " Cu",
        "Ag",
        "Au",
        "CU",
        "Co",
        "Cr",
        "Cu",
        "Fe",
        "Ni",
        "cu"
      ]
    },
    {
      "problem_id": "paper-continent",
      "gold": "Asia",
      "intermediates": [
        "China"
      ],
      "clean_completion": "Asia.",
      "jspace_completion": "Asia",
      "mean_survivor_fraction": 0.9972049689440995,
      "mean_n_excluded": 0.027950310559006212,
      "mean_delta_h_norm": 71.88674704628701,
      "n_intermediate_survivor_hits": 278,
      "n_intermediate_excluded_hits": 7,
      "n_intermediate_clean_topk_hits": 50,
      "last_pos_survivor_tokens": [
        " enam",
        " n\u00e2",
        "\u2026\\n",
        "\u3002\\",
        " \uff0c",
        " *\"",
        "...\"\\n",
        ";',",
        "\uff0c",
        "?,\\n",
        ";\\n\\n\\n\\n",
        " [",
        "...\\n",
        " (<",
        "\u00ab",
        " ...)",
        " Eis",
        "()</",
        " \\r\\n",
        ",vector",
        " enam",
        " ...\"\\n\\n",
        "?\"",
        "\u661f\u7403",
        " {\"",
        "\u5370\u5ea6",
        " Russia",
        "\u9970\u6f14",
        "ool",
        "lush",
        " enam",
        "\u6e56\u5357\u7701",
        "%f",
        ":\\n\\n\\n",
        "~",
        "!\u201d",
        " thresh",
        "qd",
        "\u5f53\u5730\u7684",
        "\">#",
        " enam",
        ",",
        "\u770b\u75c5",
        "ViewItem",
        "\u304f\u308a",
        " merging",
        "1",
        "UNET",
        "losing",
        "arry",
        " *",
        " {$",
        " '''\\n\\n",
        "[\u2026",
        ".oper",
        " \u2026",
        "\u9488",
        "5",
        " decom",
        ".Obj",
        "\uff0c",
        " ___",
        "\u3002\\",
        "\u6b63\u786e\u7684",
        " continuation",
        ";&",
        " Show",
        "\u30b9\u30bf\u30fc\u30c8",
        " \uff0c",
        "YW"
      ],
      "last_pos_excluded_tokens": [],
      "last_pos_clean_topk": [
        " Asia",
        "AS",
        "Africa",
        "As",
        "Asia",
        "Asian",
        "E",
        "East",
        "Europe",
        "\u4e9a\u6d32"
      ]
    },
    {
      "problem_id": "spider-legs",
      "gold": "8",
      "intermediates": [
        "spider"
      ],
      "clean_completion": "8",
      "jspace_completion": "8 legs.",
      "mean_survivor_fraction": 0.9968944099378882,
      "mean_n_excluded": 0.031055900621118012,
      "mean_delta_h_norm": 72.09890516056038,
      "n_intermediate_survivor_hits": 6,
      "n_intermediate_excluded_hits": 0,
      "n_intermediate_clean_topk_hits": 5,
      "last_pos_survivor_tokens": [
        " enam",
        " n\u00e2",
        "...\"\\n",
        " \uff0c",
        "\u00a0",
        "\u3002\\",
        ".\\r\\n",
        "PHA",
        " **",
        " defends",
        " park",
        " \u00ad",
        ",\\n\\n\\n\\n",
        "/pol",
        "\u533b\u62a4\u4eba\u5458",
        " [",
        "ni",
        "{})",
        " enam",
        " grantResults",
        " ...\"\\n\\n",
        "square",
        "APH",
        "\u2026\u201d",
        " enam",
        "\u9650\u5236",
        " ()=>",
        "SSION",
        ":*",
        "uby",
        " enam",
        "'.\\n",
        " __",
        "\u90fd\u8ba4\u4e3a",
        "`,\\n",
        " wed",
        " SAVE",
        "angelo",
        " \\",
        "structor",
        "rant",
        "UNET",
        " ...\"\\n\\n",
        "\u4e92",
        " permanent",
        " cert",
        "ToObject",
        "rolled",
        "\u5b89\u5168\u4e8b\u6545",
        " agreement",
        " enam",
        " \\",
        " reading",
        "lops",
        " Attributes",
        " Sinatra",
        "lify",
        " Int",
        " responses",
        "Laura",
        " \ufffd",
        "\u2026\u201d\\n\\n",
        "\u2026\\n\\n\\n",
        " ___",
        " $",
        "sons",
        "*)\\n",
        "\u4fd8",
        "\u6b63\u786e\u7684",
        " pitches"
      ],
      "last_pos_excluded_tokens": [],
      "last_pos_clean_topk": [
        "6",
        "8",
        "Eight",
        "Fact",
        "Six",
        "Sp",
        "Spider",
        "The",
        "eight",
        "\u516b"
      ]
    }
  ]
}
```

## Per-problem traces

### super-populous-capital
prompt: Fact: The capital city of the most populous country in the world is
gold: 'Beijing'  intermediates: ['China']
clean→ 'Beijing.'
J-abl→ 'Paris is the capital city of the most populous country in the world, which is'
aggregates: survivors=9.98/10.00 (frac=0.998)  excluded_dirs/step=0.02  ‖Δh‖=70.65  steps_with_any_excl=8/329
intermediate hits: survivor=297 excluded=7 clean_topk=50
last-pos clean top-10: ['Be', 'Berlin', 'China', 'London', 'New', 'Paris', 'Tok', 'Wars', 'Washington', '北京']
last-pos active J-dirs by layer ([S]=survivor ablated, [X]=excluded):
  L27 surv=10/10 ‖Δh‖=28.67 | 0:' enam'[S](c=6.78), 1:' nâ'[S](c=10.11), 2:'…\\n'[S]*G(c=13.24), 3:'。\\'[S](c=8.83), 4:' ，'[S](c=7.03), 5:'江苏省'[S](c=10.85), 6:'\xa0'[S](c=8.37), 7:' *"'[S](c=9.33), 8:'，'[S](c=5.31), 9:'…\\n\\n\\n'[S](c=8.21)
  L28 surv=10/10 ‖Δh‖=25.94 | 0:'-*'[S](c=10.79), 1:'.";\\n'[S]*G(c=9.22), 2:'«'[S](c=9.71), 3:'cycl'[S](c=10.43), 4:'：\\n'[S]*G(c=5.63), 5:' ['[S](c=6.57), 6:'”—'[S](c=3.91), 7:'…\\n\\n'[S](c=7.97), 8:'.('[S](c=7.49), 9:' Hawai'[S](c=7.61)
  L29 surv=10/10 ‖Δh‖=24.58 | 0:' enam'[S](c=1.53), 1:'...\\n\\n\\n'[S](c=7.99), 2:'":\\n'[S]*G(c=8.46), 3:'?"'[S](c=6.55), 4:'德国'[S](c=7.84), 5:' $'[S](c=5.95), 6:'依'[S](c=14.58), 7:'”;'[S](c=6.24), 8:'**,'[S](c=3.54), 9:' Ukraine'[S](c=7.92)
  L30 surv=10/10 ‖Δh‖=29.53 | 0:'":\\r\\n'[S](c=10.29), 1:' ...)'[S](c=10.64), 2:' {"'[S](c=10.36), 3:'">#'[S](c=10.52), 4:' _'[S](c=5.30), 5:'{\\n'[S]*G(c=9.41), 6:'..."\\n'[S]*G(c=5.42), 7:'าน'[S](c=14.72), 8:'`,\\n'[S]*G(c=7.68), 9:' ..."\\n\\n'[S](c=3.25)
  L31 surv=10/10 ‖Δh‖=34.10 | 0:' ..."\\n\\n'[S](c=4.05), 1:' PPP'[S](c=11.81), 2:'棵树'[S](c=14.03), 3:"'\\n\\n\\n"[S](c=9.02), 4:'):\\r\\n'[S](c=10.71), 5:' controls'[S](c=12.99), 6:' ...\\n\\n'[S](c=10.55), 7:' Blogger'[S](c=11.33), 8:' \xa0 \xa0 \xa0 \xa0'[S](c=6.96), 9:'logfile'[S](c=12.57)
  L32 surv=10/10 ‖Δh‖=45.94 | 0:' enam'[S](c=5.59), 1:'�'[S](c=11.11), 2:'邮'[S](c=18.80), 3:'不明'[S](c=17.91), 4:' […]'[S](c=8.96), 5:' Int'[S](c=17.81), 6:' \\'[S](c=5.88), 7:'atee'[S](c=18.19), 8:'毗'[S](c=18.78), 9:'Ti'[S](c=12.85)
  L33 surv=10/10 ‖Δh‖=40.84 | 0:' enam'[S](c=5.27), 1:' �'[S](c=8.34), 2:'，'[S](c=8.12), 3:'…\\n\\n\\n'[S](c=12.81), 4:',\\n\\n\\n'[S](c=13.67), 5:' sy'[S](c=17.23), 6:' ton'[S](c=18.32), 7:' _('[S](c=8.62), 8:"')["[S](c=11.07), 9:'交换'[S](c=17.82)
top survivors (all pos): ' enam':199, '”—':62, '，':52, '..."\\n':51, ' ，':51, ' **':49, ' ..."\\n\\n':47, ' \\':45, '<|endoftext|>':38, '…\\n':35
top excluded (all pos): '.\\n':4, ':\\n':3, ',':1
intermediate hit detail (first 20):
  clean_topk clean pos=1 @ 'system' → '\\n' (~China)
  clean_topk clean pos=2 @ '\\n' → 'A' (~China)
  clean_topk clean pos=2 @ '\\n' → 'In' (~China)
  clean_topk clean pos=3 @ 'Answer' → '\\n' (~China)
  clean_topk clean pos=3 @ 'Answer' → ' a' (~China)
  clean_topk clean pos=3 @ 'Answer' → ' in' (~China)
  clean_topk clean pos=3 @ 'Answer' → ':\\n' (~China)
  clean_topk clean pos=4 @ ' with' → ' a' (~China)
  clean_topk clean pos=7 @ ' entity' → ' in' (~China)
  clean_topk clean pos=7 @ ' entity' → '.\\n' (~China)
  clean_topk clean pos=10 @ ' phrase' → ' in' (~China)
  clean_topk clean pos=10 @ ' phrase' → '.\\n' (~China)
  clean_topk clean pos=11 @ ' only' → '\\n' (~China)
  clean_topk clean pos=11 @ ' only' → '.\\n' (~China)
  clean_topk clean pos=11 @ ' only' → ' in' (~China)
  clean_topk clean pos=12 @ '.' → ' In' (~China)
  clean_topk clean pos=14 @ ' line' → '.\\n' (~China)
  clean_topk clean pos=15 @ ',' → ' in' (~China)
  clean_topk clean pos=17 @ ' explanation' → ':\\n' (~China)
  clean_topk clean pos=17 @ ' explanation' → '\\n' (~China)

### amazon-language
prompt: Fact: The language spoken in the country where the Amazon River ends is
gold: 'Portuguese'  intermediates: ['Brazil']
clean→ 'Spanish.'
J-abl→ 'Spanish.'
aggregates: survivors=9.97/10.00 (frac=0.997)  excluded_dirs/step=0.03  ‖Δh‖=71.08  steps_with_any_excl=10/329
intermediate hits: survivor=9 excluded=1 clean_topk=11
last-pos clean top-10: [' Portuguese', ' Spanish', 'Brazil', 'Esp', 'French', 'Port', 'Span', 'Spanish', '葡萄牙', '西班牙']
last-pos active J-dirs by layer ([S]=survivor ablated, [X]=excluded):
  L27 surv=10/10 ‖Δh‖=26.79 | 0:' enam'[S](c=4.71), 1:'。\\'[S](c=11.89), 2:' nâ'[S](c=8.71), 3:'…\\n'[S](c=11.41), 4:' ，'[S](c=6.15), 5:'...\\n\\n'[S](c=9.07), 6:'..."\\n'[S](c=6.35), 7:'，'[S](c=5.20), 8:'医学'[S](c=9.32), 9:'江苏省'[S](c=8.54)
  L28 surv=10/10 ‖Δh‖=27.92 | 0:' ()\\n\\n'[S](c=12.17), 1:'」\\n\\n'[S](c=9.29), 2:';\\n\\n\\n\\n'[S](c=9.94), 3:' enam'[S](c=1.40), 4:' Jean'[S](c=11.09), 5:'航海'[S](c=9.84), 6:'..."\\n'[S](c=5.07), 7:" `'"[S](c=8.70), 8:'\xa0'[S](c=7.05), 9:'；'[S](c=8.51)
  L29 surv=10/10 ‖Δh‖=30.30 | 0:'\xa0'[S](c=10.21), 1:' ..."\\n\\n'[S](c=4.17), 2:' Brazilian'[S]*(c=12.05), 3:'眼界'[S](c=13.24), 4:'?"'[S](c=7.61), 5:'":\\n'[S](c=9.19), 6:'\xa0\\n'[S](c=8.44), 7:'construction'[S](c=13.31), 8:'...\\n\\n\\n'[S](c=7.38), 9:' """\\n\\n'[S](c=5.45)
  L30 surv=10/10 ‖Δh‖=33.50 | 0:']\\n\\n\\n'[S](c=12.70), 1:' ...)'[S](c=12.11), 2:' "{{'[S](c=12.02), 3:'}@'[S](c=12.61), 4:',...\\n\\n'[S](c=10.03), 5:' penet'[S](c=16.40), 6:'<|endoftext|>'[S](c=7.16), 7:' \\'[S](c=4.63), 8:'–'[S](c=4.70), 9:' **'[S](c=6.78)
  L31 surv=10/10 ‖Δh‖=40.55 | 0:'"\\n\\n'[S](c=8.64), 1:' subjects'[S](c=14.80), 2:"';"[S](c=10.30), 3:'棵树'[S](c=13.98), 4:' #'[S](c=11.25), 5:'如何看待'[S](c=13.96), 6:'publication'[S](c=12.32), 7:'竣'[S](c=16.72), 8:'块'[S](c=12.35), 9:' Savannah'[S](c=11.86)
  L32 surv=10/10 ‖Δh‖=33.68 | 0:' ..."\\n\\n'[S](c=9.34), 1:'."['[S](c=11.86), 2:' \\'[S](c=9.19), 3:' enam'[S](c=2.14), 4:' {$'[S](c=12.80), 5:'�'[S](c=7.15), 6:' \\'[S](c=7.35), 7:' \\n'[S](c=13.01), 8:' [['[S](c=9.80), 9:' proxies'[S](c=16.81)
  L33 surv=10/10 ‖Δh‖=60.39 | 0:' �'[S](c=12.34), 1:' **'[S](c=23.37), 2:' *'[S](c=16.79), 3:'=wx'[S](c=23.60), 4:' FB'[S](c=23.48), 5:'…\\n\\n\\n'[S](c=14.68), 6:'    '[S](c=10.78), 7:' (`'[S](c=15.83), 8:'接受'[S](c=23.09), 9:'asive'[S](c=20.90)
top survivors (all pos): ' enam':196, ' **':57, '”—':55, ' ，':54, '，':53, '..."\\n':52, ' ..."\\n\\n':52, ' \\':49, '<|endoftext|>':43, '\xa0':40
top excluded (all pos): '.\\n':5, ':\\n':3, ' Brazilian':1, ',':1
intermediate hit detail (first 20):
  clean_topk clean pos=2 @ '\\n' → 'A' (~Brazil)
  clean_topk clean pos=3 @ 'Answer' → ' a' (~Brazil)
  clean_topk clean pos=4 @ ' with' → ' a' (~Brazil)
  clean_topk clean pos=25 @ ':' → ' A' (~Brazil)
  clean_topk clean pos=29 @ ' in' → ' Brazil' (~Brazil)
  clean_topk clean pos=32 @ ' where' → ' I' (~Brazil)
  clean_topk clean pos=32 @ ' where' → ' a' (~Brazil)
  clean_topk clean pos=37 @ ' is' → ' Brazilian' (~Brazil)
  clean_topk clean pos=37 @ ' is' → ' a' (~Brazil)
  clean_topk clean pos=43 @ '<think>' → '\\r' (~Brazil)
  clean_topk clean pos=46 @ '\\n\\n' → 'Brazil' (~Brazil)
  active_survivor L29 pos=28 @ ' spoken' → '|R' (~Brazil) c=8.98
  active_survivor L29 pos=30 @ ' the' → ' Brazilian' (~Brazil) c=8.90
  active_survivor L29 pos=34 @ ' Amazon' → ' Brazilian' (~Brazil) c=12.19
  active_survivor L29 pos=35 @ ' River' → ' Brazilian' (~Brazil) c=13.33
  active_excluded L29 pos=37 @ ' is' → ' Brazilian' (~Brazil) c=14.11
  active_survivor L29 pos=46 @ '\\n\\n' → ' Brazilian' (~Brazil) c=12.05
  active_survivor L30 pos=34 @ ' Amazon' → '-ra' (~Brazil) c=14.28
  active_survivor L32 pos=26 @ ' The' → ':r' (~Brazil) c=12.05
  active_survivor L33 pos=14 @ ' line' → '%i' (~Brazil) c=21.41

### super-smallest-continent
prompt: Fact: The smallest country in the world is located on the continent of
gold: 'Europe'  intermediates: ['Vatican']
clean→ 'Europe.'
J-abl→ 'Europe'
aggregates: survivors=9.97/10.00 (frac=0.997)  excluded_dirs/step=0.03  ‖Δh‖=71.68  steps_with_any_excl=10/329
intermediate hits: survivor=290 excluded=8 clean_topk=43
last-pos clean top-10: ['Africa', 'Ant', 'Asia', 'Atlantic', 'E', 'EU', 'Eu', 'Euro', 'Europe', 'European']
last-pos active J-dirs by layer ([S]=survivor ablated, [X]=excluded):
  L27 surv=10/10 ‖Δh‖=26.98 | 0:' enam'[S](c=6.75), 1:' nâ'[S](c=9.11), 2:'。\\'[S](c=11.09), 3:'…\\n'[S]*(c=9.28), 4:' ，'[S](c=6.49), 5:'\xa0'[S](c=10.97), 6:' *"'[S](c=10.08), 7:'..."\\n'[S]*(c=6.44), 8:'\xa0'[S](c=7.26), 9:':\\n'[S]*(c=5.76)
  L28 surv=10/10 ‖Δh‖=22.80 | 0:';</'[S](c=7.80), 1:'«'[S](c=9.48), 2:' �'[S](c=4.47), 3:' enam'[S](c=1.08), 4:' (<'[S](c=6.90), 5:'**\\n\\n'[S](c=5.95), 6:' park'[S](c=8.08), 7:'博物馆'[S](c=8.46), 8:' physicians'[S](c=7.16), 9:'asury'[S](c=8.73)
  L29 surv=10/10 ‖Δh‖=28.13 | 0:'星球'[S](c=12.41), 1:'conditions'[S](c=11.23), 2:' ()=>'[S](c=8.50), 3:'**,'[S](c=4.04), 4:'莲花'[S](c=8.82), 5:' \\n'[S]*(c=7.94), 6:'的部分'[S](c=11.84), 7:'。\\'[S](c=3.55), 8:'!</'[S](c=7.49), 9:' Fib'[S](c=8.46)
  L30 surv=10/10 ‖Δh‖=31.51 | 0:' enam'[S](c=1.97), 1:' nest'[S](c=15.55), 2:':\\n\\n\\n'[S](c=10.81), 3:' :'[S](c=11.14), 4:'*/,\\n'[S]*(c=9.44), 5:' __'[S](c=6.26), 6:'pane'[S](c=11.61), 7:' buffer'[S](c=11.35), 8:'?</'[S](c=7.22), 9:'`\\n\\n'[S](c=7.72)
  L31 surv=10/10 ‖Δh‖=38.83 | 0:'1'[S](c=7.18), 1:','[S](c=5.73), 2:' eating'[S](c=13.35), 3:' clich'[S](c=15.27), 4:'利润率'[S](c=15.25), 5:'くり'[S](c=16.82), 6:'.”\\n\\n\\n\\n'[S](c=6.99), 7:'…\\n\\n\\n\\n'[S](c=7.99), 8:' merging'[S](c=11.98), 9:'obar'[S](c=15.31)
  L32 surv=10/10 ‖Δh‖=49.35 | 0:' enam'[S](c=2.45), 1:'.oper'[S](c=22.95), 2:' [['[S](c=11.56), 3:' oper'[S](c=18.63), 4:' *'[S](c=11.33), 5:'.Obj'[S](c=18.95), 6:'报'[S](c=15.73), 7:'aden'[S](c=15.24), 8:'十字'[S](c=14.62), 9:' stack'[S](c=15.33)
  L33 surv=10/10 ‖Δh‖=49.67 | 0:'；'[S](c=23.28), 1:'…\\n\\n\\n'[S](c=15.37), 2:'ZR'[S](c=20.70), 3:'，'[S](c=7.94), 4:'attr'[S](c=21.49), 5:' �'[S](c=6.80), 6:'。\\'[S](c=8.95), 7:'*)\\n'[S]*(c=16.09), 8:' inversion'[S](c=16.57), 9:'ȃ'[S](c=8.32)
top survivors (all pos): ' enam':210, '”—':57, ' ，':51, '，':49, ' **':49, '..."\\n':42, ' \\':41, '<|endoftext|>':40, ' ..."\\n\\n':40, '\xa0':36
top excluded (all pos): '.\\n':5, ':\\n':3, ',':1, ' \\n\\n':1
intermediate hit detail (first 20):
  clean_topk clean pos=1 @ 'system' → '\\n' (~Vatican)
  clean_topk clean pos=2 @ '\\n' → 'A' (~Vatican)
  clean_topk clean pos=3 @ 'Answer' → '\\n' (~Vatican)
  clean_topk clean pos=3 @ 'Answer' → ' a' (~Vatican)
  clean_topk clean pos=3 @ 'Answer' → ':\\n' (~Vatican)
  clean_topk clean pos=4 @ ' with' → ' a' (~Vatican)
  clean_topk clean pos=4 @ ' with' → ' an' (~Vatican)
  clean_topk clean pos=7 @ ' entity' → '.\\n' (~Vatican)
  clean_topk clean pos=10 @ ' phrase' → '.\\n' (~Vatican)
  clean_topk clean pos=11 @ ' only' → '\\n' (~Vatican)
  clean_topk clean pos=11 @ ' only' → '.\\n' (~Vatican)
  clean_topk clean pos=14 @ ' line' → '.\\n' (~Vatican)
  clean_topk clean pos=17 @ ' explanation' → ':\\n' (~Vatican)
  clean_topk clean pos=17 @ ' explanation' → '\\n' (~Vatican)
  clean_topk clean pos=17 @ ' explanation' → '.\\n' (~Vatican)
  clean_topk clean pos=19 @ '<|im_end|>' → '\\n' (~Vatican)
  clean_topk clean pos=19 @ '<|im_end|>' → ' \\n' (~Vatican)
  clean_topk clean pos=19 @ '<|im_end|>' → '  \\n' (~Vatican)
  clean_topk clean pos=21 @ '<|im_start|>' → '\\n' (~Vatican)
  clean_topk clean pos=22 @ 'user' → "'\\n" (~Vatican)

### atomic-26-symbol
prompt: Fact: The chemical symbol for the element with atomic number 26 is
gold: 'Fe'  intermediates: ['iron']
clean→ 'Fe'
J-abl→ 'Fe'
aggregates: survivors=9.97/10.00 (frac=0.997)  excluded_dirs/step=0.03  ‖Δh‖=71.07  steps_with_any_excl=11/336
intermediate hits: survivor=309 excluded=7 clean_topk=41
last-pos clean top-10: [' Fe', 'Co', 'Cr', 'FE', 'Fe', 'Iron', 'Ni', 'Nick', 'iron', '铁']
last-pos active J-dirs by layer ([S]=survivor ablated, [X]=excluded):
  L27 surv=10/10 ‖Δh‖=29.24 | 0:' enam'[S](c=5.64), 1:' nâ'[S](c=11.34), 2:'。\\'[S](c=10.72), 3:'…\\n'[S]*(c=12.15), 4:' ，'[S](c=9.69), 5:' **'[S](c=6.90), 6:' listeners'[S](c=10.53), 7:'-\\r\\n'[S](c=9.18), 8:'..."\\n'[S]*(c=5.07), 9:'».'[S](c=8.36)
  L28 surv=10/10 ‖Δh‖=23.20 | 0:'”\\n'[S]*(c=7.03), 1:' **'[S](c=6.70), 2:' park'[S](c=9.50), 3:',\\n\\n\\n\\n'[S](c=7.16), 4:' Gujarat'[S](c=8.93), 5:' ..."\\n\\n'[S](c=3.02), 6:'{})'[S](c=6.10), 7:'plash'[S](c=8.84), 8:'，“'[S](c=7.59), 9:';\\n\\n\\n\\n'[S](c=6.38)
  L29 surv=10/10 ‖Δh‖=33.68 | 0:'德国'[S](c=15.13), 1:' permission'[S](c=14.51), 2:'常德'[S](c=12.59), 3:'\xa0'[S](c=8.45), 4:'政治'[S](c=10.33), 5:'pers'[S](c=13.05), 6:' Busty'[S](c=4.53), 7:' â'[S](c=6.56), 8:'**,'[S](c=3.84), 9:'ール'[S](c=10.43)
  L30 surv=10/10 ‖Δh‖=38.19 | 0:' **'[S](c=8.90), 1:'源'[S](c=16.58), 2:'—"'[S](c=11.34), 3:'网站建设'[S](c=13.08), 4:' —\\n\\n'[S](c=10.56), 5:' …\\n\\n'[S](c=9.38), 6:' buffer'[S]G(c=11.64), 7:'quam'[S](c=16.30), 8:"')."[S](c=7.69), 9:'immel'[S](c=12.05)
  L31 surv=10/10 ‖Δh‖=47.75 | 0:' enam'[S](c=2.93), 1:' picks'[S](c=20.68), 2:'�'[S](c=14.77), 3:' pav'[S](c=13.99), 4:'cken'[S](c=18.46), 5:' gradients'[S](c=13.71), 6:'este'[S](c=15.05), 7:'lev'[S](c=18.36), 8:'~,'[S](c=12.25), 9:'isers'[S](c=13.63)
  L32 surv=10/10 ‖Δh‖=45.17 | 0:'�'[S](c=8.49), 1:' contracting'[S](c=17.07), 2:' replication'[S](c=17.70), 3:' Ness'[S](c=15.31), 4:' compuls'[S](c=18.18), 5:' conten'[S](c=16.39), 6:' {$'[S](c=9.89), 7:'他们的'[S](c=16.42), 8:' “'[S](c=11.29), 9:'…”\\n\\n'[S](c=5.75)
  L33 surv=10/10 ‖Δh‖=54.39 | 0:'…\\n\\n\\n'[S](c=17.58), 1:'，'[S](c=9.32), 2:' _('[S](c=12.47), 3:' ，'[S](c=9.12), 4:' lact'[S](c=22.15), 5:'compose'[S](c=25.73), 6:'锌'[S](c=19.06), 7:'伪装'[S](c=19.27), 8:' cath'[S](c=20.70), 9:' enam'[S](c=2.32)
top survivors (all pos): ' enam':203, '”—':59, '，':58, ' **':51, ' ，':49, '..."\\n':48, ' ..."\\n\\n':41, ' \\':41, '…\\n':39, '<|endoftext|>':37
top excluded (all pos): '.\\n':4, ':\\n':3, ',':1, '  \\n\\n':1, ' elements':1, ' number':1
intermediate hit detail (first 20):
  clean_topk clean pos=1 @ 'system' → '\\n' (~iron)
  clean_topk clean pos=3 @ 'Answer' → '\\n' (~iron)
  clean_topk clean pos=3 @ 'Answer' → ':\\n' (~iron)
  clean_topk clean pos=7 @ ' entity' → '.\\n' (~iron)
  clean_topk clean pos=10 @ ' phrase' → '.\\n' (~iron)
  clean_topk clean pos=11 @ ' only' → '\\n' (~iron)
  clean_topk clean pos=11 @ ' only' → '.\\n' (~iron)
  clean_topk clean pos=14 @ ' line' → '.\\n' (~iron)
  clean_topk clean pos=17 @ ' explanation' → ':\\n' (~iron)
  clean_topk clean pos=17 @ ' explanation' → '\\n' (~iron)
  clean_topk clean pos=17 @ ' explanation' → '.\\n' (~iron)
  clean_topk clean pos=19 @ '<|im_end|>' → '\\n' (~iron)
  clean_topk clean pos=19 @ '<|im_end|>' → ' \\n' (~iron)
  clean_topk clean pos=19 @ '<|im_end|>' → '  \\n' (~iron)
  clean_topk clean pos=21 @ '<|im_start|>' → '\\n' (~iron)
  clean_topk clean pos=22 @ 'user' → "'\\n" (~iron)
  clean_topk clean pos=22 @ 'user' → '()\\n' (~iron)
  clean_topk clean pos=22 @ 'user' → '\\n' (~iron)
  clean_topk clean pos=22 @ 'user' → '(\\n' (~iron)
  clean_topk clean pos=22 @ 'user' → '"\\n' (~iron)

### colosseum-currency
prompt: Fact: The currency used in the country where the Colosseum stands is the
gold: 'Euro'  intermediates: ['Italy']
clean→ 'Euro.'
J-abl→ 'Euro'
aggregates: survivors=9.97/10.00 (frac=0.997)  excluded_dirs/step=0.03  ‖Δh‖=68.94  steps_with_any_excl=9/350
intermediate hits: survivor=3 excluded=0 clean_topk=8
last-pos clean top-10: [' Euro', ' euro', 'E', 'EU', 'EUR', 'Euro', 'European', 'Italian', 'e', '欧元']
last-pos active J-dirs by layer ([S]=survivor ablated, [X]=excluded):
  L27 surv=10/10 ‖Δh‖=30.14 | 0:' enam'[S](c=6.30), 1:' nâ'[S](c=11.85), 2:' ，'[S](c=11.22), 3:'。\\'[S](c=9.49), 4:'…\\n'[S](c=8.61), 5:'..."\\n'[S](c=8.60), 6:'?,\\n'[S](c=11.16), 7:':`'[S](c=9.74), 8:'’.'[S](c=9.55), 9:'\xa0'[S](c=7.20)
  L28 surv=10/10 ‖Δh‖=28.58 | 0:'.";\\n'[S](c=11.22), 1:' enam'[S](c=1.59), 2:' Hawai'[S](c=10.91), 3:'**'[S](c=9.32), 4:'他们的'[S](c=10.62), 5:' divide'[S](c=9.80), 6:'!)'[S](c=8.19), 7:'飞行'[S](c=8.85), 8:' physicians'[S](c=7.88), 9:' ()\\n\\n'[S](c=7.93)
  L29 surv=10/10 ‖Δh‖=29.99 | 0:' enam'[S](c=1.65), 1:'！\\n\\n'[S](c=10.88), 2:'——'[S](c=8.87), 3:'’.'[S](c=8.97), 4:'莲花'[S](c=10.48), 5:'人性'[S](c=9.73), 6:':*'[S](c=9.05), 7:'类似的'[S](c=10.05), 8:'_TRI'[S](c=10.47), 9:' ineligible'[S](c=10.87)
  L30 surv=10/10 ‖Δh‖=30.05 | 0:' enam'[S](c=3.49), 1:'...\\n\\n'[S](c=7.22), 2:' —\\n\\n'[S](c=12.11), 3:'">#'[S](c=11.23), 4:']\\n\\n\\n'[S](c=7.65), 5:'\xad'[S](c=8.05), 6:'’'[S](c=10.43), 7:' …\\n\\n'[S](c=9.84), 8:' LZ'[S](c=11.35), 9:'—"'[S](c=10.28)
  L31 surv=10/10 ‖Δh‖=39.63 | 0:'?)'[S](c=11.54), 1:' {\\n\\n\\n\\n'[S](c=12.67), 2:'”\\n'[S](c=7.23), 3:'丈'[S](c=16.79), 4:' ()\\r\\n'[S](c=12.28), 5:')—'[S](c=12.04), 6:'中国经济'[S](c=15.09), 7:'…\\n\\n\\n\\n'[S](c=9.54), 8:' \\`'[S](c=5.87), 9:'他们的'[S](c=17.18)
  L32 surv=10/10 ‖Δh‖=38.56 | 0:' enam'[S](c=3.69), 1:'�'[S](c=10.58), 2:' Cy'[S](c=19.25), 3:' \\'[S](c=7.96), 4:' PPP'[S](c=17.16), 5:' …'[S](c=8.55), 6:' ..."\\n\\n'[S](c=5.57), 7:'医生'[S](c=13.54), 8:' \xad'[S](c=9.52), 9:'降'[S](c=16.02)
  L33 surv=10/10 ‖Δh‖=57.57 | 0:'，'[S](c=14.09), 1:' ['[S](c=15.76), 2:'...\\n\\n'[S](c=20.62), 3:'ahi'[S](c=26.72), 4:';&'[S](c=16.53), 5:'PTH'[S](c=19.96), 6:',\\n\\n\\n'[S](c=15.20), 7:' ___'[S](c=9.26), 8:' \xad'[S](c=12.56), 9:'agnost'[S](c=23.98)
top survivors (all pos): ' enam':221, '”—':59, '，':56, ' ，':52, '..."\\n':51, ' \\':49, ' ..."\\n\\n':48, '<|endoftext|>':46, ' **':45, '\xa0':36
top excluded (all pos): '.\\n':4, ':\\n':3, ',':1, '  \\n\\n':1
intermediate hit detail (first 20):
  clean_topk clean pos=2 @ '\\n' → 'A' (~Italy)
  clean_topk clean pos=3 @ 'Answer' → ' a' (~Italy)
  clean_topk clean pos=4 @ ' with' → ' a' (~Italy)
  clean_topk clean pos=25 @ ':' → ' A' (~Italy)
  clean_topk clean pos=32 @ ' where' → ' I' (~Italy)
  clean_topk clean pos=32 @ ' where' → ' a' (~Italy)
  clean_topk clean pos=32 @ ' where' → ' A' (~Italy)
  clean_topk clean pos=32 @ ' where' → ' T' (~Italy)
  active_survivor L33 pos=2 @ '\\n' → '&t' (~Italy) c=18.80
  active_survivor L33 pos=7 @ ' entity' → '&t' (~Italy) c=17.81
  active_survivor L33 pos=14 @ ' line' → '%i' (~Italy) c=21.41

### planet-3-moons
prompt: Fact: The number of natural moons orbiting the planet third from the Sun is
gold: '1'  intermediates: ['Earth']
clean→ '1'
J-abl→ '1000000000000000'
aggregates: survivors=9.97/10.00 (frac=0.997)  excluded_dirs/step=0.03  ‖Δh‖=69.45  steps_with_any_excl=12/343
intermediate hits: survivor=3 excluded=0 clean_topk=10
last-pos clean top-10: ['0', '1', '2', '3', '4', '5', '6', '9', 'Fact', 'The']
last-pos active J-dirs by layer ([S]=survivor ablated, [X]=excluded):
  L27 surv=10/10 ‖Δh‖=25.80 | 0:' enam'[S](c=6.32), 1:' nâ'[S](c=8.21), 2:'。\\'[S](c=9.58), 3:'..."\\n'[S](c=11.05), 4:' ，'[S](c=8.69), 5:'\xa0'[S](c=9.38), 6:'，'[S](c=5.59), 7:'.\\r\\n'[S](c=7.16), 8:' {})\\n'[S](c=7.62), 9:' **'[S](c=6.37)
  L28 surv=10/10 ‖Δh‖=22.13 | 0:' \\'[S](c=4.66), 1:' park'[S](c=8.89), 2:' enam'[S](c=1.04), 3:';\\n\\n\\n\\n'[S](c=7.15), 4:' _'[S](c=5.29), 5:'滋养'[S](c=9.83), 6:',\\n\\n\\n\\n'[S](c=6.52), 7:' snow'[S](c=9.05), 8:' **'[S](c=5.32), 9:' Omaha'[S](c=7.69)
  L29 surv=10/10 ‖Δh‖=21.35 | 0:' enam'[S](c=1.36), 1:' ()=>'[S](c=7.18), 2:' Iranians'[S](c=7.88), 3:' ..."\\n\\n'[S](c=2.90), 4:'”—'[S](c=3.37), 5:'`]('[S](c=7.63), 6:' waiting'[S](c=10.04), 7:' <'[S](c=6.08), 8:'实用'[S](c=8.75), 9:'！'[S](c=6.88)
  L30 surv=10/10 ‖Δh‖=25.18 | 0:' **'[S](c=6.29), 1:'�'[S](c=7.28), 2:':\\n\\n\\n'[S](c=6.96), 3:' enam'[S](c=1.00), 4:'):'[S](c=6.71), 5:'个项目'[S](c=11.82), 6:'tones'[S](c=9.46), 7:' \\'[S](c=3.18), 8:'最后'[S](c=10.35), 9:' historians'[S](c=10.00)
  L31 surv=9/10 ‖Δh‖=24.36 | 0:'iates'[S](c=11.03), 1:'1'[X]G(c=3.97), 2:' ...\\n'[S](c=9.05), 3:' `{'[S](c=6.81), 4:'.”\\n\\n\\n\\n'[S](c=4.87), 5:'ores'[S](c=11.15), 6:'?",'[S](c=5.78), 7:' ()\\r\\n'[S](c=6.86), 8:' circ'[S](c=9.42), 9:'<|endoftext|>'[S](c=4.92)
  L32 surv=10/10 ‖Δh‖=28.59 | 0:' enam'[S](c=5.38), 1:'”—'[S](c=8.77), 2:' \xad'[S](c=10.01), 3:'�'[S](c=5.97), 4:' \xad'[S](c=8.69), 5:' copper'[S](c=12.84), 6:' \\'[S](c=5.61), 7:'；'[S](c=10.42), 8:'."['[S](c=6.66), 9:'ali'[S](c=12.31)
  L33 surv=10/10 ‖Δh‖=46.31 | 0:'，'[S](c=15.92), 1:' ___'[S](c=10.67), 2:'…\\n\\n\\n'[S](c=13.60), 3:' —'[S](c=16.46), 4:' tags'[S](c=17.92), 5:'qi'[S](c=15.19), 6:')#'[S](c=12.80), 7:'泄漏'[S](c=16.45), 8:'意识'[S](c=14.76), 9:" '''\\r\\n"[S](c=10.63)
top survivors (all pos): ' enam':211, '，':55, '”—':55, '..."\\n':49, ' ，':49, ' ..."\\n\\n':49, ' \\':48, '\xa0':47, ' **':46, '<|endoftext|>':43
top excluded (all pos): '.\\n':4, ':\\n':3, '1':2, ',':1, '  \\n\\n':1, '行星':1
intermediate hit detail (first 20):
  clean_topk clean pos=2 @ '\\n' → 'A' (~Earth)
  clean_topk clean pos=3 @ 'Answer' → ' a' (~Earth)
  clean_topk clean pos=4 @ ' with' → ' a' (~Earth)
  clean_topk clean pos=25 @ ':' → ' A' (~Earth)
  clean_topk clean pos=32 @ 'ing' → ' a' (~Earth)
  clean_topk clean pos=32 @ 'ing' → ' Earth' (~Earth)
  clean_topk clean pos=33 @ ' the' → ' Earth' (~Earth)
  clean_topk clean pos=34 @ ' planet' → ' Earth' (~Earth)
  clean_topk clean pos=39 @ ' is' → ' a' (~Earth)
  clean_topk clean pos=45 @ '<think>' → '\\r' (~Earth)
  active_survivor L32 pos=26 @ ' The' → ':r' (~Earth) c=12.37
  active_survivor L33 pos=2 @ '\\n' → '&t' (~Earth) c=18.80
  active_survivor L33 pos=7 @ ' entity' → '&t' (~Earth) c=17.81

### carnival-ocean
prompt: Fact: The ocean on the coast of the country where Carnival is most famously celebrated is the
gold: 'Atlantic'  intermediates: ['Brazil']
clean→ 'Atlantic Ocean.'
J-abl→ 'Atlantic Ocean'
aggregates: survivors=9.97/10.00 (frac=0.997)  excluded_dirs/step=0.03  ‖Δh‖=68.48  steps_with_any_excl=10/357
intermediate hits: survivor=5 excluded=0 clean_topk=12
last-pos clean top-10: [' Atlantic', ' Caribbean', 'A', 'At', 'Atl', 'Atlantic', 'Car', 'English', 'North', 'Spanish']
last-pos active J-dirs by layer ([S]=survivor ablated, [X]=excluded):
  L27 surv=10/10 ‖Δh‖=29.38 | 0:' enam'[S](c=6.05), 1:'。\\'[S](c=12.35), 2:'…\\n'[S]G(c=10.94), 3:' nâ'[S](c=6.34), 4:' ，'[S](c=7.46), 5:'医学'[S](c=13.66), 6:'\xa0'[S](c=12.04), 7:'”—'[S](c=5.40), 8:'，'[S](c=5.63), 9:'\xa0'[S](c=7.93)
  L28 surv=10/10 ‖Δh‖=26.95 | 0:";'\\n"[S]G(c=13.32), 1:' enam'[S](c=1.75), 2:'：\\n'[S]G(c=6.88), 3:' matchmaking'[S](c=10.33), 4:'航海'[S](c=9.87), 5:',\\n\\n\\n\\n'[S](c=8.81), 6:' \\'[S](c=5.55), 7:';</'[S](c=7.55), 8:' –'[S](c=9.44), 9:'",\\n\\n'[S](c=6.22)
  L29 surv=10/10 ‖Δh‖=28.29 | 0:' enam'[S](c=1.84), 1:'."\\n'[S]G(c=6.75), 2:' Russia'[S](c=10.77), 3:' …\\n\\n'[S](c=11.34), 4:'\xa0'[S](c=8.78), 5:'...\\n\\n\\n'[S](c=8.80), 6:'?"'[S](c=8.06), 7:' ());\\n'[S]G(c=9.68), 8:'square'[S](c=11.18), 9:'*.'[S](c=8.32)
  L30 surv=10/10 ‖Δh‖=34.27 | 0:':\\n\\n\\n'[S](c=13.67), 1:' ...)'[S](c=12.47), 2:'–'[S](c=5.68), 3:' ..."\\n\\n'[S](c=4.33), 4:' spindle'[S](c=12.23), 5:'`,\\n'[S]G(c=8.84), 6:' semester'[S](c=11.40), 7:'也只能'[S](c=14.33), 8:"')\\n\\n\\n"[S](c=10.30), 9:'—"'[S](c=10.51)
  L31 surv=10/10 ‖Δh‖=40.40 | 0:' #'[S](c=18.60), 1:'paired'[S](c=15.18), 2:'姊妹'[S](c=14.50), 3:' ..."\\n\\n'[S](c=4.27), 4:'。\\'[S](c=5.47), 5:'eea'[S](c=14.43), 6:"...'"[S](c=9.97), 7:' reason'[S](c=14.80), 8:','[S](c=4.77), 9:' tasks'[S](c=15.60)
  L32 surv=10/10 ‖Δh‖=48.73 | 0:' enam'[S](c=1.95), 1:'/\\n\\n\\n\\n'[S](c=14.21), 2:'解答'[S](c=17.40), 3:'idis'[S](c=26.69), 4:'bull'[S](c=15.23), 5:' declarations'[S](c=14.97), 6:' matlab'[S](c=17.41), 7:' (&'[S](c=12.30), 8:' shadow'[S](c=14.01), 9:' \\'[S](c=6.54)
  L33 surv=10/10 ‖Δh‖=58.08 | 0:'，'[S](c=15.26), 1:' ，'[S](c=12.24), 2:' �'[S](c=10.02), 3:' \xad'[S](c=17.37), 4:'nu'[S](c=23.18), 5:' walker'[S](c=23.16), 6:'enus'[S](c=21.86), 7:';&'[S](c=16.79), 8:'旨'[S](c=23.98), 9:' **'[S](c=13.44)
top survivors (all pos): ' enam':216, '”—':62, '，':60, ' ，':57, ' \\':54, '\xa0':52, '..."\\n':50, ' **':49, ' ..."\\n\\n':49, '<|endoftext|>':39
top excluded (all pos): '.\\n':4, ':\\n':3, '...\\n':1, ',':1, '  \\n\\n':1
intermediate hit detail (first 20):
  clean_topk clean pos=2 @ '\\n' → 'A' (~Brazil)
  clean_topk clean pos=3 @ 'Answer' → ' a' (~Brazil)
  clean_topk clean pos=4 @ ' with' → ' a' (~Brazil)
  clean_topk clean pos=25 @ ':' → ' A' (~Brazil)
  clean_topk clean pos=31 @ ' of' → ' a' (~Brazil)
  clean_topk clean pos=34 @ ' where' → ' I' (~Brazil)
  clean_topk clean pos=34 @ ' where' → ' a' (~Brazil)
  clean_topk clean pos=34 @ ' where' → ' A' (~Brazil)
  clean_topk clean pos=36 @ ' is' → ' a' (~Brazil)
  clean_topk clean pos=40 @ ' is' → ' a' (~Brazil)
  clean_topk clean pos=47 @ '<think>' → '\\r' (~Brazil)
  clean_topk clean pos=50 @ '\\n\\n' → 'A' (~Brazil)
  active_survivor L29 pos=32 @ ' the' → ' Brazilian' (~Brazil) c=8.50
  active_survivor L29 pos=36 @ ' is' → ' Brazilian' (~Brazil) c=11.78
  active_survivor L32 pos=26 @ ' The' → ':r' (~Brazil) c=12.23
  active_survivor L32 pos=30 @ ' coast' → ':r' (~Brazil) c=13.17
  active_survivor L33 pos=14 @ ' line' → '%i' (~Brazil) c=21.41

### atomic-29-symbol
prompt: Fact: The chemical symbol for the element with atomic number 29 is
gold: 'Cu'  intermediates: ['copper']
clean→ 'Cu'
J-abl→ 'Cu'
aggregates: survivors=9.97/10.00 (frac=0.997)  excluded_dirs/step=0.03  ‖Δh‖=71.12  steps_with_any_excl=11/336
intermediate hits: survivor=5 excluded=0 clean_topk=9
last-pos clean top-10: [' Cu', 'Ag', 'Au', 'CU', 'Co', 'Cr', 'Cu', 'Fe', 'Ni', 'cu']
last-pos active J-dirs by layer ([S]=survivor ablated, [X]=excluded):
  L27 surv=10/10 ‖Δh‖=30.55 | 0:' enam'[S](c=4.40), 1:' nâ'[S](c=11.97), 2:'。\\'[S](c=13.00), 3:'…\\n'[S](c=13.32), 4:' ，'[S](c=11.05), 5:' **'[S](c=6.49), 6:'-\\r\\n'[S](c=11.36), 7:'..."\\n'[S](c=5.39), 8:'，'[S](c=4.44), 9:' listeners'[S](c=9.00)
  L28 surv=10/10 ‖Δh‖=25.88 | 0:' park'[S](c=10.63), 1:' `"'[S](c=9.56), 2:' **'[S](c=7.09), 3:'，“'[S](c=8.37), 4:'体育馆'[S](c=8.81), 5:'他们的'[S](c=9.15), 6:';\\n\\n\\n\\n'[S](c=6.94), 7:'”\\n'[S](c=5.02), 8:' Gujarat'[S](c=8.02), 9:'빙'[S](c=6.79)
  L29 surv=10/10 ‖Δh‖=28.01 | 0:' ..."\\n\\n'[S](c=4.94), 1:'德国'[S](c=10.65), 2:'\xa0'[S](c=8.03), 3:'…\\n'[S](c=8.98), 4:':\\n\\n\\n'[S](c=9.73), 5:'].\\n\\n'[S](c=8.78), 6:'常德'[S](c=9.57), 7:' permission'[S](c=10.28), 8:'骑行'[S](c=8.98), 9:'——'[S](c=7.11)
  L30 surv=10/10 ‖Δh‖=40.43 | 0:'ób'[S](c=13.83), 1:'—"'[S](c=11.17), 2:'юр'[S](c=13.21), 3:'-hot'[S](c=18.00), 4:' **'[S](c=7.26), 5:':\\n\\n\\n'[S](c=9.43), 6:' Ridley'[S](c=11.34), 7:' mate'[S](c=13.11), 8:'源'[S](c=13.41), 9:'mans'[S](c=14.08)
  L31 surv=10/10 ‖Δh‖=56.43 | 0:'圣诞'[S](c=26.41), 1:' enam'[S](c=2.66), 2:' Emit'[S](c=20.38), 3:' append'[S](c=20.99), 4:' memes'[S](c=17.68), 5:' picks'[S](c=19.49), 6:'…\\n\\n'[S](c=8.44), 7:'riba'[S](c=18.52), 8:'-flex'[S](c=16.83), 9:' letters'[S](c=15.23)
  L32 surv=10/10 ‖Δh‖=51.12 | 0:' enam'[S](c=6.00), 1:' …'[S](c=13.59), 2:' \\'[S](c=12.01), 3:' oper'[S](c=20.81), 4:'图'[S](c=17.95), 5:'dy'[S](c=18.49), 6:'”—'[S](c=8.56), 7:'actionDate'[S](c=18.10), 8:'.Compare'[S](c=21.04), 9:' setId'[S](c=17.43)
  L33 surv=10/10 ‖Δh‖=59.45 | 0:'，'[S](c=18.57), 1:'…\\n\\n\\n'[S](c=21.12), 2:'泄漏'[S](c=25.66), 3:' ，'[S](c=8.96), 4:' \xad'[S](c=14.88), 5:' intents'[S](c=21.69), 6:' lact'[S](c=20.74), 7:'">{'[S](c=15.20), 8:'…\\n\\n'[S](c=13.12), 9:'.CL'[S](c=21.79)
top survivors (all pos): ' enam':205, '，':57, '”—':55, ' **':52, '..."\\n':50, ' ，':49, ' \\':43, ' ..."\\n\\n':40, '…\\n':38, '<|endoftext|>':37
top excluded (all pos): '.\\n':4, ':\\n':3, ',':1, '  \\n\\n':1, ' elements':1, ' number':1
intermediate hit detail (first 20):
  clean_topk clean pos=13 @ ' One' → ' per' (~copper)
  clean_topk clean pos=14 @ ' line' → ' per' (~copper)
  clean_topk clean pos=29 @ ' for' → ' copper' (~copper)
  clean_topk clean pos=31 @ ' element' → ' copper' (~copper)
  clean_topk clean pos=38 @ ' is' → ' Co' (~copper)
  clean_topk clean pos=40 @ '\\n' → ' copper' (~copper)
  clean_topk clean pos=44 @ '<think>' → '\\r' (~copper)
  clean_topk clean pos=45 @ '\\n\\n' → ' copper' (~copper)
  clean_topk clean pos=47 @ '\\n\\n' → 'Co' (~copper)
  active_survivor L32 pos=8 @ ' or' → ' copper' (~copper) c=13.94
  active_survivor L32 pos=24 @ 'Fact' → ' copper' (~copper) c=20.73
  active_survivor L32 pos=26 @ ' The' → ':r' (~copper) c=11.43
  active_survivor L32 pos=29 @ ' for' → ':r' (~copper) c=13.53
  active_survivor L32 pos=45 @ '\\n\\n' → ':r' (~copper) c=13.30

### paper-continent
prompt: Fact: The continent where the country that invented paper is located is
gold: 'Asia'  intermediates: ['China']
clean→ 'Asia.'
J-abl→ 'Asia'
aggregates: survivors=9.97/10.00 (frac=0.997)  excluded_dirs/step=0.03  ‖Δh‖=71.89  steps_with_any_excl=9/322
intermediate hits: survivor=278 excluded=7 clean_topk=50
last-pos clean top-10: [' Asia', 'AS', 'Africa', 'As', 'Asia', 'Asian', 'E', 'East', 'Europe', '亚洲']
last-pos active J-dirs by layer ([S]=survivor ablated, [X]=excluded):
  L27 surv=10/10 ‖Δh‖=26.05 | 0:' enam'[S](c=6.52), 1:' nâ'[S](c=8.34), 2:'…\\n'[S]*(c=9.39), 3:'。\\'[S](c=7.83), 4:' ，'[S](c=6.37), 5:' *"'[S](c=11.60), 6:'..."\\n'[S]*(c=7.49), 7:";',"[S](c=9.00), 8:'，'[S](c=5.13), 9:'?,\\n'[S]*(c=8.67)
  L28 surv=10/10 ‖Δh‖=24.86 | 0:';\\n\\n\\n\\n'[S](c=9.47), 1:' ['[S](c=7.42), 2:'...\\n'[S]*(c=7.35), 3:' (<'[S](c=7.40), 4:'«'[S](c=8.08), 5:' ...)'[S](c=7.58), 6:' Eis'[S](c=8.63), 7:'()</'[S](c=5.26), 8:' \\r\\n'[S](c=7.53), 9:',vector'[S](c=9.03)
  L29 surv=10/10 ‖Δh‖=29.08 | 0:' enam'[S](c=2.18), 1:' ..."\\n\\n'[S](c=3.57), 2:'?"'[S](c=7.30), 3:'星球'[S](c=10.07), 4:' {"'[S](c=8.77), 5:'印度'[S](c=9.25), 6:' Russia'[S](c=7.22), 7:'饰演'[S](c=11.21), 8:'ool'[S](c=15.36), 9:'lush'[S](c=9.82)
  L30 surv=10/10 ‖Δh‖=36.64 | 0:' enam'[S](c=2.50), 1:'湖南省'[S](c=12.25), 2:'%f'[S](c=13.34), 3:':\\n\\n\\n'[S](c=10.66), 4:'~'[S](c=11.33), 5:'!”'[S](c=11.97), 6:' thresh'[S](c=13.28), 7:'qd'[S](c=11.84), 8:'当地的'[S](c=14.27), 9:'">#'[S](c=10.13)
  L31 surv=10/10 ‖Δh‖=36.10 | 0:' enam'[S](c=2.85), 1:','[S](c=7.72), 2:'看病'[S](c=12.89), 3:'ViewItem'[S](c=15.00), 4:'くり'[S](c=14.72), 5:' merging'[S](c=11.60), 6:'1'[S](c=4.33), 7:'UNET'[S](c=13.95), 8:'losing'[S](c=11.55), 9:'arry'[S](c=11.95)
  L32 surv=10/10 ‖Δh‖=51.80 | 0:' *'[S](c=16.00), 1:' {$'[S](c=14.88), 2:" '''\\n\\n"[S](c=13.79), 3:'[…'[S](c=13.82), 4:'.oper'[S](c=20.65), 5:' …'[S](c=9.70), 6:'针'[S](c=18.36), 7:'5'[S](c=14.61), 8:' decom'[S](c=18.05), 9:'.Obj'[S](c=20.67)
  L33 surv=10/10 ‖Δh‖=57.90 | 0:'，'[S](c=11.83), 1:' ___'[S](c=13.65), 2:'。\\'[S](c=11.58), 3:'正确的'[S](c=26.10), 4:' continuation'[S](c=23.64), 5:';&'[S](c=16.75), 6:' Show'[S](c=22.15), 7:'スタート'[S](c=22.50), 8:' ，'[S](c=7.40), 9:'YW'[S](c=17.90)
top survivors (all pos): ' enam':195, '，':52, ' ，':51, '”—':51, ' ..."\\n\\n':50, ' \\':49, ' **':46, '..."\\n':45, '<|endoftext|>':43, '\xa0':34
top excluded (all pos): '.\\n':4, ':\\n':3, ',':1, '  \\n\\n':1
intermediate hit detail (first 20):
  clean_topk clean pos=1 @ 'system' → '\\n' (~China)
  clean_topk clean pos=2 @ '\\n' → 'A' (~China)
  clean_topk clean pos=2 @ '\\n' → 'In' (~China)
  clean_topk clean pos=3 @ 'Answer' → '\\n' (~China)
  clean_topk clean pos=3 @ 'Answer' → ' a' (~China)
  clean_topk clean pos=3 @ 'Answer' → ' in' (~China)
  clean_topk clean pos=3 @ 'Answer' → ':\\n' (~China)
  clean_topk clean pos=4 @ ' with' → ' a' (~China)
  clean_topk clean pos=7 @ ' entity' → ' in' (~China)
  clean_topk clean pos=7 @ ' entity' → '.\\n' (~China)
  clean_topk clean pos=10 @ ' phrase' → ' in' (~China)
  clean_topk clean pos=10 @ ' phrase' → '.\\n' (~China)
  clean_topk clean pos=11 @ ' only' → '\\n' (~China)
  clean_topk clean pos=11 @ ' only' → '.\\n' (~China)
  clean_topk clean pos=11 @ ' only' → ' in' (~China)
  clean_topk clean pos=12 @ '.' → ' In' (~China)
  clean_topk clean pos=14 @ ' line' → '.\\n' (~China)
  clean_topk clean pos=15 @ ',' → ' in' (~China)
  clean_topk clean pos=17 @ ' explanation' → ':\\n' (~China)
  clean_topk clean pos=17 @ ' explanation' → '\\n' (~China)

### spider-legs
prompt: Fact: The number of legs on the animal that spins webs is
gold: '8'  intermediates: ['spider']
clean→ '8'
J-abl→ '8 legs.'
aggregates: survivors=9.97/10.00 (frac=0.997)  excluded_dirs/step=0.03  ‖Δh‖=72.10  steps_with_any_excl=10/322
intermediate hits: survivor=6 excluded=0 clean_topk=5
last-pos clean top-10: ['6', '8', 'Eight', 'Fact', 'Six', 'Sp', 'Spider', 'The', 'eight', '八']
last-pos active J-dirs by layer ([S]=survivor ablated, [X]=excluded):
  L27 surv=10/10 ‖Δh‖=29.22 | 0:' enam'[S](c=6.64), 1:' nâ'[S](c=8.99), 2:'..."\\n'[S](c=10.23), 3:' ，'[S](c=8.03), 4:'\xa0'[S](c=10.92), 5:'。\\'[S](c=6.16), 6:'.\\r\\n'[S](c=9.65), 7:'PHA'[S](c=11.97), 8:' **'[S](c=7.26), 9:' defends'[S](c=10.61)
  L28 surv=10/10 ‖Δh‖=26.97 | 0:' park'[S](c=11.48), 1:' \xad'[S](c=10.14), 2:',\\n\\n\\n\\n'[S](c=7.96), 3:'/pol'[S](c=9.23), 4:'医护人员'[S](c=7.35), 5:' ['[S](c=5.22), 6:'ni'[S](c=12.31), 7:'{})'[S](c=5.57), 8:' enam'[S](c=0.87), 9:' grantResults'[S](c=8.78)
  L29 surv=10/10 ‖Δh‖=28.78 | 0:' ..."\\n\\n'[S](c=4.81), 1:'square'[S](c=13.25), 2:'APH'[S](c=11.53), 3:'…”'[S](c=7.35), 4:' enam'[S](c=1.20), 5:'限制'[S](c=10.65), 6:' ()=>'[S](c=7.09), 7:'SSION'[S](c=10.29), 8:':*'[S](c=7.56), 9:'uby'[S](c=10.64)
  L30 surv=10/10 ‖Δh‖=31.29 | 0:' enam'[S](c=2.05), 1:"'.\\n"[S](c=8.91), 2:' __'[S](c=7.44), 3:'都认为'[S](c=12.08), 4:'`,\\n'[S](c=8.75), 5:' wed'[S](c=13.12), 6:' SAVE'[S](c=13.22), 7:'angelo'[S](c=11.56), 8:' \\'[S](c=4.48), 9:'structor'[S](c=10.69)
  L31 surv=10/10 ‖Δh‖=48.14 | 0:'rant'[S](c=19.43), 1:'UNET'[S](c=17.12), 2:' ..."\\n\\n'[S](c=4.40), 3:'互'[S](c=14.87), 4:' permanent'[S](c=16.69), 5:' cert'[S](c=14.66), 6:'ToObject'[S](c=17.45), 7:'rolled'[S](c=16.06), 8:'安全事故'[S](c=12.94), 9:' agreement'[S](c=13.38)
  L32 surv=10/10 ‖Δh‖=55.33 | 0:' enam'[S](c=2.87), 1:' \\'[S](c=10.45), 2:' reading'[S](c=25.63), 3:'lops'[S](c=20.88), 4:' Attributes'[S](c=18.51), 5:' Sinatra'[S](c=20.06), 6:'lify'[S](c=16.20), 7:' Int'[S](c=20.66), 8:' responses'[S](c=16.05), 9:'Laura'[S](c=12.54)
  L33 surv=10/10 ‖Δh‖=58.09 | 0:' �'[S](c=13.72), 1:'…”\\n\\n'[S](c=11.76), 2:'…\\n\\n\\n'[S](c=16.99), 3:' ___'[S](c=11.84), 4:' $'[S](c=14.14), 5:'sons'[S](c=26.07), 6:'*)\\n'[S](c=18.74), 7:'俘'[S](c=21.28), 8:'正确的'[S](c=24.07), 9:' pitches'[S](c=18.67)
top survivors (all pos): ' enam':199, '”—':53, '..."\\n':50, ' ..."\\n\\n':47, '，':46, ' ，':46, ' **':46, '\xa0':44, ' \\':42, '<|endoftext|>':41
top excluded (all pos): '.\\n':4, ':\\n':3, ',':1, '  \\n\\n':1, ' \\n\\n':1
intermediate hit detail (first 20):
  clean_topk clean pos=31 @ ' the' → ' spider' (~spider)
  clean_topk clean pos=33 @ ' that' → "'s" (~spider)
  clean_topk clean pos=42 @ '<think>' → '\\r' (~spider)
  clean_topk clean pos=45 @ '\\n\\n' → 'Sp' (~spider)
  clean_topk clean pos=45 @ '\\n\\n' → 'Spider' (~spider)
  active_survivor L32 pos=0 @ '<|im_start|>' → '�s' (~spider) c=244.96
  active_survivor L32 pos=17 @ ' explanation' → 'DER' (~spider) c=20.14
  active_survivor L32 pos=25 @ ':' → '�s' (~spider) c=8.72
  active_survivor L32 pos=26 @ ' The' → ':r' (~spider) c=12.37
  active_survivor L33 pos=14 @ ' line' → '%i' (~spider) c=21.41
  active_survivor L33 pos=27 @ ' number' → '|i' (~spider) c=15.08


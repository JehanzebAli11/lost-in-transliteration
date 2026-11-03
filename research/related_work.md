# Related work — "Lost in Transliteration"

Compiled 2026-11-03. **Verification rule:** every entry in the main list was opened in this session (arXiv abstract page, ACL Anthology page, Hugging Face model card, GitHub README or vendor blog), and its title, authors and date or venue were checked against that page. Venues are given only when the opened page states them (for example in the arXiv "Comments" field or on the ACL Anthology page). Otherwise the entry is cited by its arXiv ID. Entries dated 2026 are recent preprints, and most of them have not been peer-reviewed.

Format for each entry: **citation**, the link that was opened, *Found:* one sentence on the result, *Ours:* one sentence on how this study differs.

---

## Key novelty question: has anyone already tested Urdu or Roman Urdu for prompt injection or jailbreaks?

**Searches run (2026-11-03):**
- arXiv API (abstract and title search): `"Roman Urdu" AND jailbreak`, `Urdu AND jailbreak`, `Urdu AND "prompt injection"`, `Urdu AND safety AND LLM`, `Urdu AND adversarial`, `Urdu AND LLM AND evaluation`, `"Roman Urdu"` (all), `transliteration AND jailbreak`, `transliteration AND safety`, `transliterated AND LLM`, `romanized AND jailbreak`, `romanized AND safety`, `romanized AND LLM AND attack`, `Hinglish AND jailbreak`, `Hinglish AND safety`, `Hindi AND jailbreak`, `Bengali AND jailbreak`, `"code-mixed" AND jailbreak`, `"code-switching" AND jailbreak`, `guardrail AND "code-mixed"`, `multilingual AND "prompt injection"`, `"prompt injection" AND multilingual AND classifier`, `"prompt injection" AND "non-English"`, `"prompt injection" AND translation`, `"prompt injection" AND language AND low-resource`, `"Prompt Guard" AND multilingual`, `"tool poisoning" AND MCP`, `"South Asian" AND safety AND LLM`.
- Full-text checks: I downloaded the arXiv HTML of 25 of the closest papers and searched each for "Urdu" and "romaniz/romanis/transliterat" (details in the entries below).
- GitHub repository search (`gh search repos`): "roman urdu prompt injection", "roman urdu jailbreak", "urdu jailbreak", "urdu prompt injection", "hinglish jailbreak".
- Semantic Scholar API (`/graph/v1/paper/search`) for "Roman Urdu jailbreak", "Urdu prompt injection", "transliteration jailbreak", "Hinglish jailbreak" and "romanized jailbreak". **Every request returned HTTP 429 (rate-limited, no API key), so this source was not actually searched.** ACL Anthology has no search API I could use, so it was checked only by opening specific papers.

- Resource list check (2026-11-03): <https://github.com/traversaal-ai/urdu-llm-resources> ("Resources for Urdu Large Language Models", 330-line README). It lists no prompt-injection, jailbreak or red-teaming dataset. The only safety-related entries are Roman Urdu hate-speech/toxicity data, for example <https://github.com/haroonshakeel/roman_urdu_hate_speech>, and Roman Urdu sentiment and transliteration resources.

**What I found:**
1. **No paper found tests prompt injection (direct or indirect) written in Urdu script, Roman Urdu or code-switched Urdu-English.** None evaluates dedicated injection classifiers (Prompt Guard, ProtectAI, etc.) or MCP scanners on Urdu variants, and none fine-tunes a detector for them. Every multilingual prompt-injection paper I found uses other languages: Bangla (MIPIAD), Tamil/Tanglish (Madhusudhanan), Turkish/Russian/Chinese (Uysal et al.), or Japanese/Chinese/Arabic (Theocharopoulos et al.).
2. **Urdu has appeared in jailbreak (harmful-request) studies.** The closest is **IndicJR (Pattnayak & Chowdhuri, EACL 2026 Industry)**. It covers 12 South Asian languages, Urdu among them, with native, romanized and mixed-script variants made by machine transliteration (AI4Bharat). It found that romanization *reduced* jailbreak success (by about 0.34 on average) under its JSON-contract track. That cuts against our hypothesis, and the write-up should discuss it. Urdu also appears among the translated languages in Shen et al. (*The Language Barrier*; GPT-4 harmful rate 27% for Urdu) and in Song et al. (*Multilingual Blending*). In all three, the target is harmful-content jailbreaks against chat LLMs, not prompt injection or classifiers. None uses natively typed, informal Roman Urdu.
3. **Closest work on Urdu script variants in safety classification:** *Ghaib in Translation* (Zehra et al., 2026) compares Nastaliq Urdu, Roman Urdu, English and code-switched Urdu-English when LLMs act as hate-speech classifiers. It reports 2.4–9.9% "Missed-in-Urdu" rates. Its conditions are nearly the same as our four, but the task is hate speech, not injection.
4. **Closest work on "a small fine-tuned classifier closes the script gap":** *BanglaShield* (Kabir et al., 2026). For Bangla, a compact fine-tuned encoder reaches 87.8 / 84.4 macro-F1 on native and romanized prompts, beating LLM classifiers and multilingual guard models by more than 40 points. *BanglaVeilGuard* (Hassan & Hossain, 2026) is similar. Both target harmful-prompt safety in Bangla, not prompt injection in Urdu. Our Phase 2 result should be framed as a replication or extension of this idea to injection detection in Urdu, not as a new idea.
5. **Grey literature:** a GitHub repo (Mahad7836/garak-multilingual-promptinj-eval) runs a small English vs machine-translated Urdu garak `promptinject` evaluation on Mistral-7B (64 attempts per language). It is the only direct Urdu prompt-injection evaluation I could open. It uses Urdu script only, one model, and no classifiers or Roman Urdu. A second repo, "AfraSadat/english-urdu-prompt-injection-dataset", came up in search, but I could not open its contents (see *Could not verify*).

**Bottom line:** to the best of these searches, this is the first study to measure **prompt-injection detection by dedicated classifiers and MCP scanners** across **Urdu script, Roman Urdu and code-switched Urdu-English** as separate conditions. It is not the first to show that Urdu or romanized South Asian text affects LLM safety (see IndicJR, Language Barrier, Ghaib in Translation), and it is not the first to show that a small fine-tuned encoder closes a script gap (see BanglaShield). Semantic Scholar could not be searched, so a paper indexed only there may have been missed.

---

## 1. Multilingual and low-resource-language jailbreaks

**Yong, Menghini & Bach (2023).** *Low-Resource Languages Jailbreak GPT-4.* NeurIPS 2023 Workshop SoLaR (Best Paper). arXiv:2310.02446. <https://arxiv.org/abs/2310.02446>
*Found:* Translating AdvBench prompts into low-resource languages (Zulu, Scots Gaelic, Hmong, Guarani) bypassed GPT-4's safeguards 79% of the time, while mid- and high-resource languages (including Bengali and Hindi) were far less effective. Urdu was not tested.
*Ours:* We test prompt injection with benign canary goals rather than harmful requests, we score dedicated injection classifiers and MCP scanners as well as LLMs, and we treat Urdu script, Roman Urdu and code-switched text as separate conditions.

**Deng, Zhang, Pan & Bing (2023).** *Multilingual Jailbreak Challenges in Large Language Models* (MultiJail). ICLR 2024. arXiv:2310.06474. <https://arxiv.org/abs/2310.06474>
*Found:* Unsafe-output rates rise as language resources fall, both for accidental non-English use and for deliberate multilingual attacks. The benchmark covers 9 languages including Bengali; Urdu appears only in their resource-level table and is not evaluated.
*Ours:* We study injection (instruction hijacking inside documents and tool descriptions), not harmful-content elicitation, and we include romanized and code-switched Urdu, which MultiJail lacks.

**Wang, Tu, Chen, Yuan et al. (2023).** *All Languages Matter: On the Multilingual Safety of Large Language Models* (XSafety). ACL 2024 Findings. arXiv:2310.00905. <https://arxiv.org/abs/2310.00905>
*Found:* Across 10 languages (including Hindi and Bengali), all four LLMs produced significantly more unsafe responses to non-English queries than to English ones.
*Ours:* We cover Urdu (absent from XSafety) in three written forms, and we measure detector recall and false positives, not only model outputs.

**Shen, Tan, Chen, Chen et al. (2024).** *The Language Barrier: Dissecting Safety Challenges of LLMs in Multilingual Contexts.* arXiv:2401.13136. <https://arxiv.org/abs/2401.13136>
*Found:* LLMs give unsafe and irrelevant responses more often to malicious prompts in lower-resource languages. Urdu is one of the low-resource languages tested (GPT-4 harmful rate 27% for Urdu vs about 1% for high-resource languages).
*Ours:* One of the few prior papers with Urdu (Nastaliq script, translated), but for harmful requests only. We add Roman Urdu and code-switched forms and test injection classifiers and indirect injection.

**Puttaparthi, Deo, Gul, Tang et al. (2023).** *Comprehensive Evaluation of ChatGPT Reliability Through Multilingual Inquiries.* arXiv:2312.10524. <https://arxiv.org/abs/2312.10524>
*Found:* Fuzzing ChatGPT with multilingual "wrapping" of malicious questions (7,892 Q&A points) causes jailbreaks, and wrapping them in prompt-injection templates raises the jailbreak probability further.
*Ours:* We isolate the effect of language and script on injection itself, using benign goals, and we test detectors rather than one chat model.

**Li, Liu, Liu, Shi et al. (2024).** *A Cross-Language Investigation into Jailbreak Attacks in Large Language Models.* arXiv:2401.16765. <https://arxiv.org/abs/2401.16765>
*Found:* A semantics-preserving multilingual jailbreak dataset breaks GPT-4 and LLaMa, and multilingual fine-tuning cut attack success by 96.2%.
*Ours:* Our mitigation is a cheap classifier fine-tuned on a small amount of Roman Urdu data rather than fine-tuning the LLM, and we measure a data-efficiency curve.

**Yoo, Yang & Lee (2024).** *Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding* (CSRT). ACL 2025. arXiv:2406.15481. <https://arxiv.org/abs/2406.15481>
*Found:* Synthetic code-switched red-teaming queries (up to 10 languages mixed) produced 46.7% more successful attacks than English across ten LLMs.
*Ours:* CSRT's code-switching is artificial multi-language mixing. Ours is natural, two-language Urdu-English mixing in Latin script, as Pakistani users type it, and it is applied to injection detection.

**Song, Huang, Zhou & Ma (2024).** *Multilingual Blending: LLM Safety Alignment Evaluation with Language Mixture.* arXiv:2407.07342. <https://arxiv.org/abs/2407.07342>
*Found:* Mixed-language query-response schemes raised safety bypass rates to 67.23% on GPT-3.5 and 40.34% on GPT-4o, with effectiveness varying by morphology and language family. Urdu is among the blended languages (native script).
*Ours:* We test single-language Urdu variants and natural Urdu-English code-switching as controlled conditions, for injection rather than harmful queries.

**Upadhayay & Behzadan (2024).** *Sandwich attack: Multi-language Mixture Adaptive Attack on LLMs.* arXiv:2404.07242. <https://arxiv.org/abs/2404.07242>
*Found:* Hiding a harmful question among questions in several low-resource languages elicited harmful or misaligned answers from Bard, Gemini Pro, LLaMA-2, GPT-3.5/4 and Claude 3 Opus.
*Ours:* We do not search for adversarial language mixtures; we ask whether ordinary everyday Urdu forms already slip past injection defences.

**Banerjee, Chatterjee, Kumar, Layek et al. (2025).** *Attributional Safety Failures in Large Language Models under Code-Mixed Perturbations.* arXiv:2505.14469. <https://arxiv.org/abs/2505.14469>
*Found:* Code-mixed inputs raise attack success on open models from 9% (monolingual English) to 69%, and above 90% for Arabic and Hindi contexts; saliency-drift attribution is used to explain why.
*Ours:* Similar code-mixing hypothesis but applied to Urdu-English and to prompt-injection classifiers and indirect injection, not harmful-request refusal.

**Pattnayak & Chowdhuri (2026).** *IndicJR: A Judge-Free Benchmark of Jailbreak Robustness in South Asian Languages.* EACL 2026 Industry Track. arXiv:2602.16832. <https://arxiv.org/abs/2602.16832>
*Found:* Across 12 South Asian languages **including Urdu**, with native, romanized and mixed-script variants made by machine transliteration, English-to-Indic jailbreaks transfer strongly (Urdu/Hindi about 0.70 JSR). Romanized or mixed inputs *reduce* jailbreak success under the JSON-contract track, with romanization share and tokenization correlating with the effect.
*Ours:* **Closest prior work.** We study prompt injection with benign canaries rather than harmful jailbreaks, we evaluate injection classifiers and MCP scanners, and our Roman Urdu is generated to imitate informal typing rather than produced by rule-based transliteration. Their finding that romanization lowered attack success is a direct counter-hypothesis we must report against.

**Mondal, Mamta, Varshney, Cocarascu et al. (2026).** *IndicSafeEval: Safety Robustness of Large Language Models under Multilingual Persuasive Jailbreak Attacks.* Findings of EMNLP 2026. arXiv:2609.03781. <https://arxiv.org/abs/2609.03781>
*Found:* 7,200 persuasion-based jailbreak prompts in Hindi, Bengali, Marathi and Punjabi show that safety behaviour of open LLMs varies by language, persuasion strategy and risk category. No Urdu and no romanization (full-text check).
*Ours:* We cover Urdu and its romanized and code-switched forms, and injection rather than persuasion jailbreaks.

**Ifebi, Kong & Mehrotra (2026).** *Minionese: Comprehensive Benchmark and Mechanistic Study of Multilingual LLM Safety.* arXiv:2607.10112. <https://arxiv.org/abs/2607.10112>
*Found:* Across 18 languages and four perturbations (translation, code-switching, transliteration, translationese), transliteration vulnerability is mediated by script identity, and low-resource jailbreaks route harmful content through a subspace that misses refusal directions. No Urdu (full-text check).
*Ours:* We test whether transliteration effects carry over to injection detectors, specifically for Urdu, and our stretch probe (RQ7) asks a related representation question for injections.

**Bhatnagar (2026).** *Safety Alignment Illusion: The Cross-Lingual Safety Gap in LLMs* (INCLUDE benchmark). arXiv:2608.18131 (submitted to IEEE SLT 2026). <https://arxiv.org/abs/2608.18131>
*Found:* A 2,604-prompt bias benchmark in English, Hindi, Bengali, Marathi, Tamil and Hinglish shows that English-centric safety alignment fails to carry over to Indian languages, including romanized Hinglish.
*Ours:* We test security (injection) rather than social bias, and Roman Urdu rather than Hinglish.

---

## 2. Transliteration, romanized script and encodings; Roman Urdu as an under-resourced form

**Ghanim, Almohaimeed, Zheng, Solihin et al. (2024).** *Jailbreaking LLMs with Arabic Transliteration and Arabizi.* EMNLP 2024. arXiv:2406.18725. <https://arxiv.org/abs/2406.18725>
*Found:* AdvBench in Standard Arabic rarely jailbroke GPT-4 or Claude 3 Sonnet, but Arabic transliteration and Arabizi (Latin-script "chatspeak") did produce unsafe content.
*Ours:* The closest analogue for a romanized Perso-Arabic-script language. Arabizi is to Arabic roughly what Roman Urdu is to Urdu, but they studied jailbreaks on chat LLMs, while we study injection against classifiers, LLMs and MCP scanners.

**Yuan, Jiao, Wang, Huang et al. (2023).** *GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher* (CipherChat). ICLR 2024. arXiv:2308.06463. <https://arxiv.org/abs/2308.06463>
*Found:* Talking to GPT-4 in ciphers (Caesar, Morse, etc.) bypasses safety alignment that was trained on natural language.
*Ours:* Roman Urdu is not an obfuscation but a natural writing system used by millions; we test whether ordinary use, not deliberate encoding, already creates a gap.

**Husain, Dabre, Kumar, Gala et al. (2024).** *RomanSetu: Efficiently unlocking multilingual capabilities of Large Language Models via Romanization.* ACL 2024. arXiv:2401.14280. <https://arxiv.org/abs/2401.14280>
*Found:* Romanized text cuts token fertility by 2–4x and its embeddings sit closer to English than native-script embeddings, which improves LLM performance for non-Latin-script languages.
*Ours:* RomanSetu suggests LLMs may *understand* Roman Urdu well (so injections could succeed) while detectors trained on English or native script may not recognise it; we measure both sides.

**Khullar, Desai, Malviya, Dalmia et al. (2025).** *Script Gap: Evaluating LLM Triage on Indian Languages in Native vs Romanized Scripts in a Real World Setting.* arXiv:2512.10780. <https://arxiv.org/abs/2512.10780>
*Found:* On real maternal-health queries in five Indian languages and Nepali, LLM triage accuracy drops by up to 24 points for romanized messages.
*Ours:* Same native-vs-romanized axis, applied to security detection in Urdu rather than clinical triage.

**Zehra, Kara-Isitt, Khosla & Swift (2026).** *'Ghaib in Translation' aka Unseen Harm: Measuring Cross-Script Safety Inconsistency with 'Missed-in-Urdu' Scores in LLM Hate Speech Detection.* arXiv:2608.24191. <https://arxiv.org/abs/2608.24191>
*Found:* Five LLMs used as hate-speech classifiers on Nastaliq Urdu, Roman Urdu, English and code-switched Urdu-English data showed 15.9–31.6% label instability between Urdu and its English translation, with 2.4–9.9% "Missed-in-Urdu" harm. A census of 205 WOAH/ALW papers found zero dedicated Urdu papers.
*Ours:* **Closest prior work on Urdu script variants in safety classification.** It uses the same four-way language design, but the task is hate speech judged by LLMs. We study prompt injection judged by dedicated classifiers, LLMs and MCP scanners, and we test a fix.

**Kabir, Tariquzzaman, Rifat, Ranasinghe et al. (2026).** *BanglaShield: A Script-Robust Safety Classifier and Benchmark for Bangla Native, Romanized, and Code-Mixed Prompts.* arXiv:2610.24288. <https://arxiv.org/abs/2610.24288>
*Found:* LLM classifiers label 47.0% of unsafe native-script Bangla prompts as safe, rising to 63.4% under romanization, and multilingual guard models pass 30.3–57.3%. A compact fine-tuned encoder reaches 87.8 / 84.4 macro-F1 on native and romanized text, beating the best LLM by more than 42 points.
*Ours:* **Closest prior work for our Phase 2 idea** (a small script-robust encoder). We target prompt injection rather than harmful-prompt safety and Urdu rather than Bangla, and we measure a data-efficiency curve (how *little* Roman-Urdu data is needed) with held-out templates and styles.

**Hassan & Hossain (2026).** *BanglaVeilGuard: Cross-Script Safety Benchmarking and Lightweight Guardrails for Bangla Large Language Models.* ICCA 2026. arXiv:2608.21880. <https://arxiv.org/abs/2608.21880>
*Found:* A lightweight prompt guard with multi-view normalization across six Bangla forms (native, romanized, Banglish, code-mixed, noisy, dialectal) cut attack success from 93.8–100% to about 6.3%, at the cost of over-refusal on dialectal and noisy benign text.
*Ours:* Same "lightweight guard for romanized forms" idea but for Bangla harmful prompts; we report over-blocking on benign Roman Urdu (RQ2) as a first-class metric, motivated by exactly this cost.

**Hashmat, Naseem & Raza (2026).** *Pak3H: Evaluating the Cost of Cultural Mismatch in LLM Alignment with a Human-Contextualized Urdu Benchmark.* arXiv:2608.30065. <https://arxiv.org/abs/2608.30065>
*Found:* A human-validated, culturally adapted Urdu benchmark for helpfulness, harmlessness and honesty shows LLM performance degrading on all three in Urdu.
*Ours:* Pak3H covers alignment in Urdu script; we cover adversarial injection and include Roman Urdu.

**Khan, Shahzad & Malik (2021).** *Hate Speech Detection in Roman Urdu.* ACM TALLIP (per arXiv comment). arXiv:2108.02830. <https://arxiv.org/abs/2108.02830>
*Found:* From over 90,000 scraped tweets the authors built a 5,000-tweet Roman Urdu hate-speech corpus, claiming it was the first such study, which illustrates how thin Roman Urdu safety data is.
*Ours:* It shows Roman Urdu moderation data is scarce; we address a different safety task (injection) where, as far as we found, no Roman Urdu data exists.

**Hussain, Qasim, Mehak, Zain et al. (2025).** *Fine-Tuning Large Language Models with QLoRA for Offensive Language Detection in Roman Urdu-English Code-Mixed Text.* arXiv:2510.03683. <https://arxiv.org/abs/2510.03683>
*Found:* Translating code-mixed Roman Urdu-English to English and fine-tuning LLMs with QLoRA improved offensive-language detection, while noting that translation discards code-mixing features.
*Ours:* We classify Roman Urdu and code-switched text directly (no translate-to-English step) and for injection, not offensiveness.

**Furqan, Khaja & Habeeb (2024).** *ERUPD — English to Roman Urdu Parallel Dataset.* arXiv:2412.17562. <https://arxiv.org/abs/2412.17562>
*Found:* A 75,146-pair English–Roman Urdu parallel corpus mixing synthetic and real chat data, citing Roman Urdu's lack of standardisation, phonetic variability and code-switching as obstacles.
*Ours:* These properties are why we treat Roman Urdu as its own condition; ERUPD could serve as an external reference for how natural our generated Roman Urdu is.

**Butt, Veranasi & Neumann (2025).** *Low-Resource Transliteration for Roman-Urdu and Urdu Using Transformer-Based Models.* arXiv:2503.21530. <https://arxiv.org/abs/2503.21530>
*Found:* An m2m100-based model improves Urdu ↔ Roman Urdu transliteration on Roman-Urdu-Parl and Dakshina, noting that transliteration between the two scripts "remains underexplored".
*Ours:* It confirms Roman Urdu is under-resourced and suggests a possible defence (transliterate to Urdu script before classifying) that we do not test but could mention as future work.

---

## 3. Cross-lingual gaps in safety classifiers and guardrails

**Inan, Upasani, Chi, Rungta et al. (2023).** *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.* arXiv:2312.06674. <https://arxiv.org/abs/2312.06674>
*Found:* A Llama-2-7B model instruction-tuned on a safety taxonomy performs strongly on English prompt- and response-classification benchmarks.
*Ours:* Llama Guard targets content harms, not injection; we focus on injection-specific detectors and test them outside English.

**Kumar, Jain, Yerukola, Jiang et al. (2025).** *PolyGuard: A Multilingual Safety Moderation Tool for 17 Languages.* COLM 2025. arXiv:2504.04377. <https://arxiv.org/abs/2504.04377>
*Found:* A safety classifier trained on 1.91M samples in 17 languages (including Hindi, not Urdu) beats existing open and commercial safety classifiers by 5.5%.
*Ours:* PolyGuard shows that multilingual training data helps; we ask how *little* targeted data is needed for injection detection in an uncovered language and script.

**de Wynter, Watts, Wongsangaroonsri, Zhang et al. (2024).** *RTP-LX: Can LLMs Evaluate Toxicity in Multilingual Scenarios?* AAAI 2025. arXiv:2404.14397. <https://arxiv.org/abs/2404.14397>
*Found:* On human-transcreated toxic prompts in 28 languages, LLM judges score acceptable accuracy but agree poorly with humans and miss subtle, context-dependent harm.
*Ours:* We use objective success criteria (exact canary or secret match) rather than toxicity judgments, which avoids judge disagreement.

**Jain, Kumar, Gehman, Zhou et al. (2024).** *PolygloToxicityPrompts: Multilingual Evaluation of Neural Toxic Degeneration in Large Language Models.* COLM 2024. arXiv:2405.09373. <https://arxiv.org/abs/2405.09373>
*Found:* Across 425K naturally occurring prompts in 17 languages and over 60 LLMs, toxicity rises as language resources fall.
*Ours:* The same resource-gap pattern, tested for injection rather than toxic generation and in an under-resourced *script variant* rather than a language.

**Aakanksha, Ahmadian, Ermis, Goldfarb-Tarrant et al. (2024).** *The Multilingual Alignment Prism: Aligning Global and Local Preferences to Reduce Harm* (Aya Red-teaming). arXiv:2406.18682. <https://arxiv.org/abs/2406.18682>
*Found:* Human-annotated red-teaming prompts in several languages (including Hindi), separating global and local harms, support safety alignment across 6 languages with minimal loss of general performance.
*Ours:* We do not align a generator; we evaluate and cheaply improve an input filter.

**Friedrich, Tedeschi, Schramowski, Brack et al. (2024).** *LLMs Lost in Translation: M-ALERT uncovers Cross-Linguistic Safety Inconsistencies.* arXiv:2412.15035. <https://arxiv.org/abs/2412.15035>
*Found:* Across 75k prompts in five European languages and 39 LLMs, safety is inconsistent across languages and categories even among high-resource languages.
*Ours:* If inconsistency appears even between European languages, a larger gap is plausible for Urdu; we test that for injection.

**Yang, Dan, Li, Roth et al. (2025).** *MrGuard: A Multilingual Reasoning Guardrail for Universal LLM Safety.* arXiv:2504.15241. <https://arxiv.org/abs/2504.15241>
*Found:* A reasoning guardrail trained on synthetic multilingual data with curriculum GRPO beats baselines by more than 15% and stays robust to code-switching.
*Ours:* We test a much cheaper fix (an encoder classifier of about 280M parameters) for injection, and we measure robustness to romanization, which MrGuard does not address.

**Chua, Tan, Ge & Lee (2025).** *Lost in Localization: Building RabakBench with Human-in-the-Loop Validation to Measure Multilingual Safety Gaps.* arXiv:2507.05980. <https://arxiv.org/abs/2507.05980>
*Found:* A localized benchmark (Singlish, Chinese, Malay, Tamil) shows 13 guardrails degrade significantly on code-mixed vernaculars.
*Ours:* A parallel finding for Singapore's languages; our vernacular is Roman Urdu and code-switched Urdu-English, and the task is injection.

---

## 4. Prompt-injection fundamentals, benchmarks, and MCP tool poisoning

**Perez & Ribeiro (2022).** *Ignore Previous Prompt: Attack Techniques For Language Models.* NeurIPS 2022 ML Safety Workshop. arXiv:2211.09527. <https://arxiv.org/abs/2211.09527>
*Found:* Simple handcrafted inputs achieve goal hijacking and prompt leaking against GPT-3 (PromptInject framework).
*Ours:* Our OVERRIDE and SECRET_LEAK goals follow these two attack types, translated into three Urdu variants.

**Greshake, Abdelnabi, Mishra, Endres et al. (2023).** *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* arXiv:2302.12173. <https://arxiv.org/abs/2302.12173>
*Found:* Instructions planted in retrieved data can control LLM-integrated apps (e.g. Bing Chat), enabling data theft and worming, among other harms.
*Ours:* RQ3 uses this indirect setting (an instruction hidden in a document to be summarised), but varies the language of the hidden instruction.

**Liu, Jia, Geng, Jia et al. (2023).** *Formalizing and Benchmarking Prompt Injection Attacks and Defenses.* USENIX Security 2024. arXiv:2310.12815. <https://arxiv.org/abs/2310.12815>
*Found:* A formal framework for injection attacks, used to benchmark 5 attacks and 10 defenses across 10 LLMs and 7 tasks.
*Ours:* Their benchmark is English-only; we add language and script as an axis.

**Yi, Xie, Zhu, Kiciman et al. (2023).** *Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models* (BIPIA). KDD 2025. arXiv:2312.14197. <https://arxiv.org/abs/2312.14197>
*Found:* All tested LLMs are vulnerable to indirect injection because they cannot separate context from instructions; boundary-awareness and explicit-reminder defenses help.
*Ours:* English-only; our carrier-document design is similar but varies the injection language.

**Zhan, Liang, Ying & Kang (2024).** *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents.* Findings of ACL 2024. arXiv:2403.02691. <https://arxiv.org/abs/2403.02691> (also <https://aclanthology.org/2024.findings-acl.624/>)
*Found:* Across 1,054 cases, ReAct-prompted GPT-4 followed indirect injections 24% of the time, nearly doubling with a "hacking prompt".
*Ours:* English-only; we test language variation and include scanners, not just agents.

**Debenedetti, Zhang, Balunović, Beurer-Kellner et al. (2024).** *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.* arXiv:2406.13352. <https://arxiv.org/abs/2406.13352>
*Found:* An extensible environment with 629 security test cases shows that existing attacks break some agent security properties but not all, and that agents often fail tasks even without attacks.
*Ours:* A possible future host for Urdu-variant injections; our study is a smaller, static measurement.

**Toyer, Watkins, Mendes, Svegliato et al. (2023).** *Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game.* arXiv:2311.01011. <https://arxiv.org/abs/2311.01011>
*Found:* 126,000+ human-written injection attacks and 46,000 defenses from an online game; many models are vulnerable to extraction and hijacking, and some strategies transfer to deployed apps.
*Ours:* Crowd-sourced English attacks; ours are templated and translated into Urdu variants, plus a natively generated set.

**Schulhoff, Pinto, Khan, Bouchard et al. (2023).** *Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition.* EMNLP 2023. arXiv:2311.16119. <https://arxiv.org/abs/2311.16119>; <https://aclanthology.org/2023.emnlp-main.302/>
*Found:* 600K+ adversarial prompts from a global competition show that LLMs are easily prompt-hacked, and the paper offers a taxonomy of techniques.
*Ours:* Public English injection datasets like this may be used to bulk up English training data in Phase 2; we measure what they miss in Urdu.

**Uysal, Birinci, Mutluergil & Çetin (2026).** *An Empirical Evaluation of Prompt Injection Vulnerabilities in Large Language Models Across Multilingual and Obfuscated Attack Scenarios.* AI-SS 2026 Workshop at EDCC 2026. arXiv:2606.29602. <https://arxiv.org/abs/2606.29602>
*Found:* Testing six LLMs with malicious-artifact requests in English, Turkish, Russian and Chinese and with Base64, ROT13 and hex encodings, non-English prompts consistently got higher compliance than English.
*Ours:* Their "prompt injection" means harmful-artifact requests to chat models; ours means hijacking via embedded instructions, tested on detectors and with Urdu variants.

**Theocharopoulos, Kulkarni & Magimai.-Doss (2025).** *Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing.* arXiv:2512.23684. <https://arxiv.org/abs/2512.23684>
*Found:* Hidden injections in about 500 ICML papers changed LLM review scores for English, Japanese and Chinese injections, but Arabic injections had little effect.
*Ours:* A real example of language-dependent *indirect* injection success, and it suggests Perso-Arabic script may *weaken* injection, which is consistent with IndicJR. Our RQ3 tests this for Urdu script vs Roman Urdu.

**Madhusudhanan G (2026).** *Visible Reasoning and Indirect Prompt-Injection Monitorability Across English, Tamil, and Tanglish.* arXiv:2608.15392. <https://arxiv.org/abs/2608.15392>
*Found:* A small pilot (eight scenarios, one model) of indirect injection in English, Tamil and romanized Tanglish on Sarvam-105B gave inconclusive, direction-reversing results about whether visible reasoning helps.
*Ours:* The only indirect-injection study with a romanized South Asian condition that I found; ours is larger, covers Urdu, and includes classifiers and scanners.

**Mahad7836 (n.d., accessed 2026-11-03).** *garak-multilingual-promptinj-eval: Multilingual Prompt-Injection Evaluation with garak* (GitHub repository, not peer-reviewed). README opened at <https://raw.githubusercontent.com/Mahad7836/garak-multilingual-promptinj-eval/main/README.md>
*Found:* garak `promptinject.HijackHateHumans` on Mistral-7B gave 35.94% automated attack success in English vs 14.06% in machine-translated Urdu. Manual review of 20 attempts confirmed 15% (English) vs 0% (Urdu), and the author warns that translation can break attacks and exact-string detectors.
*Ours:* The only direct Urdu prompt-injection evaluation I found. It covers Urdu script only, one model, no classifiers, no Roman Urdu or code-switching. Its warning about translation degrading attacks supports our canary-survival checks and human-review sample.

**Beurer-Kellner & Fischer (2025).** *MCP Security Notification: Tool Poisoning Attacks.* Invariant Labs blog, 2025-04-01. <https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks>
*Found:* Disclosed "tool poisoning": hidden instructions in MCP tool descriptions can make clients such as Cursor exfiltrate data. A later update announced the MCP-Scan scanner.
*Ours:* We test whether scanners built after this disclosure catch the same poisoning when written in Urdu variants.

**Radosevich & Halloran (2025).** *MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits.* arXiv:2504.03767. <https://arxiv.org/abs/2504.03767>
*Found:* Leading LLMs can be coerced through MCP tools into malicious code execution, remote access and credential theft; the authors release MCPSafetyScanner, an agentic auditing tool.
*Ours:* We evaluate existing MCP scanners as detectors, with language as the variable.

**Wang, Gao, Wang, Liu et al. (2025).** *MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers.* arXiv:2508.14925. <https://arxiv.org/abs/2508.14925>
*Found:* On 45 real MCP servers, 353 tools and 1,348 poisoned cases, tool poisoning succeeded widely (o1-mini 72.8% ASR). More capable models were often more susceptible, and refusal rates were below 3%.
*Ours:* MCPTox measures agent susceptibility in English; we measure *scanner* detection across Urdu variants.

**Shaikh & Montaruli (2026).** *MCP Tool Poisoning in Agentic AI Frameworks: A Cross-Framework Measurement Study.* AISec '26 (co-located with ACM CCS 2026). arXiv:2611.03555. <https://arxiv.org/abs/2611.03555>
*Found:* Across 12 frameworks, 18 models and 45 scenarios, tool poisoning succeeded in 22.9% of cells, no framework had an intentional defense, and a regex description scanner rejected 20 of 45 scenarios at 7.2% false positives ("metadata scanning alone is a partial control").
*Ours:* Their regex-scanner result is English; pattern-based scanners are exactly what we expect to fail on Urdu script and Roman Urdu, and we measure that.

---

## 5. Robustness of injection and jailbreak classifiers

**Meta (2024).** *Prompt-Guard-86M model card.* Hugging Face. <https://huggingface.co/meta-llama/Prompt-Guard-86M>
*Found:* Prompt Guard is mDeBERTa-v3-base fine-tuned for jailbreak and injection detection; its multilingual evaluation covers machine-translated attacks in English, French, German, Hindi, Italian, Portuguese, Spanish and Thai.
*Ours:* Urdu is not in its evaluation languages, and romanized text is not mentioned; we measure it on Urdu script, Roman Urdu and code-switched input.

**Meta (2025).** *Llama-Prompt-Guard-2-86M model card.* Hugging Face. <https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M>
*Found:* Prompt Guard 2 86M (mDeBERTa) reports multilingual AUC .995 vs English .998, evaluated on the same language set, and warns that the 22M variant has a larger multilingual gap because no multilingual deberta-xsmall exists.
*Ours:* Vendor-reported multilingual scores exclude Urdu; we provide an independent number for an untested language and its romanization.

**ProtectAI (2024).** *deberta-v3-base-prompt-injection-v2 model card.* Hugging Face. <https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2>
*Found:* The card says the model "does not … handle non-English prompts" and is accurate for English injections.
*Ours:* We quantify that stated limitation for Urdu variants, including Roman Urdu, which uses Latin script and could partly fool an English model.

**Chennabasappa, Nikolaidis, Song, Molnar et al. (2025).** *LlamaFirewall: An open source guardrail system for building secure AI agents.* arXiv:2505.03574. <https://arxiv.org/abs/2505.03574>
*Found:* Meta's agent guardrail framework combines PromptGuard 2, agent alignment checks and code scanning as a final defensive layer for agents.
*Ours:* We test one component (PromptGuard-style detection) in a language setting the framework's evaluation does not cover.

**Hackett, Birch, Trawicki, Suri et al. (2025).** *Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems.* LLMSec 2025. arXiv:2504.11168. <https://arxiv.org/abs/2504.11168>
*Found:* Character injection (homoglyphs, zero-width characters, diacritics, etc.) and adversarial-ML word perturbations evaded six guardrails, including Azure Prompt Shield, Meta Prompt Guard and ProtectAI, sometimes 100% of the time. Prompt Guard averaged about 70% injection evasion.
*Ours:* Their evasions are deliberate perturbations; ours are natural writing forms of a language that Zehra et al. (2026) put at 246M speakers. A gap there would hit benign-looking, non-adversarial traffic too.

**Jacob, Alzahrani, Hu, Alomair et al. (2025).** *PromptShield: Deployable Detection for Prompt Injection Attacks.* ACM CODASPY 2025. arXiv:2501.15145. <https://arxiv.org/abs/2501.15145>
*Found:* A curated benchmark and fine-tuned detector deliver much better performance in the low-false-positive regime, showing that data curation matters.
*Ours:* We also emphasise false-positive rates (RQ2), and we test whether a small amount of curated Roman Urdu data is enough.

**Li & Liu (2024).** *InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models* (NotInject). arXiv:2410.22770. <https://arxiv.org/abs/2410.22770>
*Found:* Guard models over-flag benign inputs containing trigger words (accuracy near 60% on NotInject); the MOF training strategy reduces this bias.
*Ours:* Our instruction-like benign controls ("Ignore the typo…") follow NotInject's idea, extended to Urdu variants, where over-blocking may hurt everyday users.

**Li, Liu, Zhang & Xiao (2025).** *PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free.* ACL 2025 (Long), pp. 30420–30437. <https://aclanthology.org/2025.acl-long.1468/>
*Found:* The peer-reviewed version of InjecGuard, published under the name PIGuard.
*Ours:* As above; PIGuard is a candidate classifier for our Phase 1 evaluation.

**Li, Chen, Hu, Chen et al. (2024).** *GenTel-Safe: A Unified Benchmark and Shielding Framework for Defending Against Prompt Injection Attacks.* arXiv:2409.19521. <https://arxiv.org/abs/2409.19521>
*Found:* An 84,812-attack benchmark (GenTel-Bench) and a detector (GenTel-Shield) that reports state-of-the-art detection against existing guardrails.
*Ours:* Large but English- and Chinese-centric attack sets; we contribute a small, controlled Urdu-variant test set.

**Hung, Ko, Rawat, Chung et al. (2024).** *Attention Tracker: Detecting Prompt Injection Attacks in LLMs.* arXiv:2411.00348. <https://arxiv.org/abs/2411.00348>
*Found:* Specific attention heads shift from the original instruction to injected instructions (the "distraction effect"), enabling training-free detection with up to 10% AUROC gains.
*Ours:* A model-internal signal like this relates to our stretch RQ7 (linear probe); we ask whether internal signals transfer across scripts.

**Muhtadi & Tazwar (2026).** *MIPIAD: Multilingual Indirect Prompt Injection Attack Defense with Qwen -- TF-IDF Hybrid and Meta-Ensemble Learning.* arXiv:2605.07269. <https://arxiv.org/abs/2605.07269>
*Found:* On a 1.43M-sample synthetic BIPIA-based benchmark in English and Bangla, a LoRA-tuned Qwen2.5-1.5B classifier plus TF-IDF ensemble reached F1 0.92, and ensembles reduced the English-Bangla gap.
*Ours:* The closest multilingual *injection-detector* work; it covers native-script Bangla only (no romanization, no Urdu), and we additionally evaluate existing off-the-shelf detectors and MCP scanners.

**Rahman, Shahriar, Wu & Cuzzocrea (2024).** *Applying Pre-trained Multilingual BERT in Embeddings for Improved Malicious Prompt Injection Attacks Detection.* arXiv:2409.13331. <https://arxiv.org/abs/2409.13331>
*Found:* Multilingual BERT embeddings with classical classifiers improve injection detection.
*Ours:* Despite the "multilingual" encoder, evaluation is on English data; we test cross-script generalisation explicitly.

---

## Could not verify

- **AfraSadat/english-urdu-prompt-injection-dataset** (GitHub). Seen only as a name in `gh search repos "urdu prompt injection"`. The GitHub web page returned HTTP 403 through the proxy, the API refused access for this session, and no README was found at `raw.githubusercontent.com/.../{main,master}/README.md`. Its contents, date and scope are unknown. **The owner should open it manually, because it may be direct prior work (an English–Urdu prompt-injection dataset).** *Follow-up (2026-11-03):* the owner pointed to the Hugging Face collection <https://huggingface.co/collections/afrasadat/prompt-injection>, which probably belongs to the same person ("Afra Sadat"). Opened via the HF API, it contains only two bookmarked papers: Rahman et al. 2024 (arXiv:2409.13331, already cited above) and Ayub & Majumdar 2024, *Embedding-based classifiers can detect prompt injection attacks* (arXiv:2410.22284). The HF account has no public datasets, models or spaces. So the collection is not the dataset, and the GitHub repo's contents remain unverified. *Update (2026-11-03):* the owner downloaded the file (`Dataset.xlsx`, last modified by "Afra Sadat", 2026-10-15). It is a parallel English/Urdu-script set of 5,016 + 5,016 short direct-injection and jailbreak prompts (1,516 attacks and 3,500 benign per language, about 50% duplicates), with no Roman Urdu and no code-switching. It is the closest prior *dataset*. Its licence and how the Urdu was produced are still unknown. Analysis and usage plan: `research/external_datasets.md`.
- **Avan7699/multilingual_llm_jailbreaks** (GitHub; description: "English/Hindi/Hinglish jailbreaks and detection"). The README contains only a link to a PDF report, which I did not open, so its findings are unverified.
- **Semantic Scholar searches** (all five queries listed above) returned HTTP 429 every time, so results indexed only by Semantic Scholar or the ACL Anthology (for example workshop papers on Roman Urdu safety) may be missing.
- **Venues not shown on the pages I opened, so not claimed:** Greshake et al. (often cited as AISec 2023), Tensor Trust (often cited as ICLR 2024), AgentDojo (often cited as NeurIPS 2024 Datasets & Benchmarks), Shen et al. *Language Barrier*. These are cited above by arXiv ID only.
- **Meta Prompt Guard / Llama Guard "multilingual weakness" evaluations by third parties:** apart from Hackett et al. (character evasion, not languages) and BanglaShield (guard models on Bangla), I did not find and open an independent paper that specifically measures Prompt Guard on non-English languages.

---

## Positioning

Prior work shows that LLM safety degrades in low-resource languages, code-switched text and transliteration (Yong et al.; Deng et al.; Yoo et al.; Ghanim et al.; Minionese). Urdu has appeared in jailbreak benchmarks (Shen et al.; IndicJR) and, with Roman Urdu and code-switching, in hate-speech classification (*Ghaib in Translation*). However, these searches found no study of prompt injection in Urdu variants beyond a small English-versus-Urdu repository. Our contribution is narrower than "first multilingual injection study". We measure dedicated injection classifiers, frontier and open models under indirect injection, and MCP tool-poisoning scanners on matched Urdu-script, Roman Urdu and code-switched items, using benign, exactly scorable goals. IndicJR found that romanization lowered jailbreak success, so our hypothesis is not assured. Our fix echoes BanglaShield's small script-robust encoder. What is new is applying it to injection and mapping how little Roman Urdu data is needed. Our data is machine-generated, so the natively written set is our main robustness check.

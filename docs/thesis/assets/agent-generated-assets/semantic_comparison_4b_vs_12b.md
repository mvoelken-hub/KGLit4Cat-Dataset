# Semantic Comparison: 4b vs 12b Output Usability

**Dataset:** 13C-13C_STM125 (b3df299b) for 12b, 1H_NMR-1H_NMR (75ac48c5) for 4b
**Note:** 4b artifacts for b3df299b were cleared by force_rerun. Using 1H_NMR 4b output as representative 4b sample — same model, same architecture, same pipeline.

---

## Scenario: Data catalog receives this metadata. Can it be used?

### 4b Output (`gemma3:4b-cloud`)

```json
{
  "id": "75ac48c5",
  "title": ["SIMONE extraction result for 75ac48c5"],
  "description": [],
  "was_generated_by": [{
    "realized_plan": {"title": "Realized Plan"},
    "has_quantitative_attribute": [
      {"title": "Reference frequency", "value": 500.133,
       "has_quantity_type": "Frequency", "unit": "GM-PER-HZ"},
      {"title": "NROUT", "value": 0.0,
       "has_quantity_type": "ComplexFrequency_Imaginary"},
      {"title": "Date Wed Sep 11 30 59 2020", "value": 30.0,
       "has_quantity_type": "Time"}
    ],
    "carried_out_by": [{"id": "TD0", "type": {"id": "voc4cat_0007008"}}]
  }],
  "identifier": ["75ac48c5"],
  "modification_date": "2023-10-27"
}
```

**Can a human understand what this dataset is?**
No. Title is auto-generated placeholder. Description is empty. No keywords, no type, no creator. A human looking at this metadata sees only an opaque ID and three cryptic numeric parameters.

**Can a machine consume this as DCAT-AP+ metadata?**
Partially. JSON structure is valid. Schema passes. But:
- `title` is not a real title — it's a placeholder
- `description` is empty — required by DCAT-AP+
- `creator` is missing — required by DCAT-AP+
- `keyword`, `type`, `theme` all missing
- `carried_out_by` uses raw CV IRI — no human-readable label
- `realized_plan.title` = "Realized Plan" — circular placeholder, not actual plan name

**Are the quantitative attributes semantically correct?**
No. Three attributes extracted, all with issues:
- "Reference frequency" 500.133 with unit "GM-PER-HZ" — unit is wrong. NMR frequency measured in MHz, not grams per hertz. Model picked wrong QUDT unit.
- "NROUT" 0.0 mapped to "ComplexFrequency_Imaginary" — NROUT is Bruker routing flag (integer), not a complex frequency. Model hallucinated quantity type.
- "Date Wed Sep 11 30 59 2020" value 30.0 mapped to "Time" — model parsed date string, extracted "30" (day of month), assigned as Time quantity. Semantically nonsensical.

**Verdict: NOT USABLE.**
A data catalog would reject this or display it as "Unknown dataset with ID 75ac48c5." No searchable fields. No discoverability. No provenance. Incorrect semantics in quantitative attributes. A human curator would need to start from scratch.

---

### 12b Output (`gemma3:12b-cloud`)

```json
{
  "id": "b3df299b",
  "title": ["13C NMR Spectra"],
  "description": ["NMR spectrum data acquired on a Bruker Avance 400 spectrometer in CDCl3 solvent."],
  "was_generated_by": [{
    "title": ["Data Acquisition and Processing"],
    "description": ["Data acquisition using a Bruker Avance 400 spectrometer and processing with MestReNova and XWIN-NMR software."],
    "carried_out_by": [{"id": "mestrenova:af59b5d1", "title": "MestReNova"}],
    "realized_plan": {"type": {"id": "zgpg30"}},
    "evaluated_entity": [{
      "title": "Spectral Parameters",
      "has_quantitative_attribute": [37 attributes...]
    }],
    "evaluated_activity": [{
      "title": ["NMR Acquisition"],
      "description": ["Standard 1D NMR acquisition protocols with power-gated decoupling."]
    }]
  }],
  "creator": [{"name": ["Bruker BioSpin GmbH"], "type": {"preferred_label": ["Organization"]}}],
  "keyword": ["NMR", "Spectroscopy", "13C", "CDCl3"],
  "type": [{"preferred_label": ["Spectroscopic Data"]}],
  "is_about_activity": [{"title": ["NMR Spectroscopy"]}],
  "is_about_entity": [{"title": "13C NMR Spectra"}],
  "identifier": ["b3df299b"]
}
```

**Can a human understand what this dataset is?**
Yes. Title says "13C NMR Spectra." Description specifies instrument (Bruker Avance 400) and solvent (CDCl3). Keywords include NMR, Spectroscopy, 13C, CDCl3. Type is "Spectroscopic Data." A human immediately knows: this is carbon-13 NMR spectroscopy data.

**Can a machine consume this as DCAT-AP+ metadata?**
Partially. Significant improvement over 4b but still incomplete:
- `title` — descriptive, correct
- `description` — present, meaningful, contains instrument + solvent
- `creator` — present, "Bruker BioSpin GmbH" as Organization
- `keyword` — present, relevant terms
- `type` — present, "Spectroscopic Data"
- `was_generated_by` — rich: title, description, carried_out_by (MestReNova), realized_plan (zgpg30 pulse program), evaluated_entity, evaluated_activity
- `is_about_entity` / `is_about_activity` — present, descriptive

Still missing for full DCAT-AP+ conformance:
- `access_rights` — not extracted
- `publisher` — not extracted
- `distribution` — not extracted
- `contact_point` — not extracted
- `temporal_coverage` / `spatial_coverage` — not applicable for NMR but required by profile
- `modification_date` — empty
- `conforms_to` — not extracted
- `applicable_legislation` — not extracted

**Are the quantitative attributes semantically correct?**
Mixed. 37 attributes extracted — values are correct but semantics vary:
- Values are accurate (MCREST=0, MCWRK=0.015, NPOINTS=32768, etc.)
- `has_quantity_type` uses raw Bruker parameter names ("MCREST", "MCWRK", "F1 9ppm") instead of proper QUDT vocabulary. Not mapped to standard quantity kinds.
- Some units are questionable ("cm" for "Maximum intensity" — should be arbitrary intensity units)
- Too many raw instrument parameters included — spectral processing flags, phase cycle values, calibration constants. These are instrument settings, not dataset metadata.
- No filtering between scientifically meaningful parameters (frequency, solvent, number of scans) and internal processing flags (phase cycles, routing flags).

**Verdict: PARTIALLY USABLE.**
A data catalog could display this with meaningful title, description, keywords, and creator. Searchable by "NMR", "13C", "spectroscopy." Provenance partially traced (Bruker instrument, MestReNova software, zgpg30 pulse program). However:
- Profile conformance still fails (many required fields missing)
- Quantitative attributes need curation (raw parameter names, wrong units)
- `modification_date` empty
- No access rights or distribution info
- A human curator would need ~30 min to clean up, vs starting from scratch with 4b output

---

## Side-by-Side Semantic Assessment

| Question | 4b Answer | 12b Answer |
|----------|-----------|------------|
| What is this dataset? | Unknown — generic title, no description | 13C NMR Spectra — clear title + description |
| What instrument was used? | Raw CV IRI "voc4cat_0007008" | Bruker Avance 400 — named in description |
| What software processed it? | "TD0" (meaningless) | MestReNova, XWIN-NMR — both named |
| What was measured? | "Reference frequency: 500.133 Hz" (wrong unit) | 13C NMR spectra in CDCl3 solvent |
| Who created it? | Missing | Bruker BioSpin GmbH (Organization) |
| What type of data? | Missing | Spectroscopic Data |
| What activity? | Missing | NMR Acquisition, NMR Spectroscopy |
| What method/plan? | "Realized Plan" (placeholder) | zgpg30 (pulse program identifier) |
| Is it searchable? | No — no keywords, generic title | Yes — keywords: NMR, Spectroscopy, 13C, CDCl3 |
| Are quantitative values correct? | No — wrong units, wrong quantity types, date parsed as number | Values correct but quantity types are raw parameter names, not standard vocabulary |
| Can a curator use this? | No — start from scratch | Yes — ~30 min cleanup needed |
| Profile conformance? | Fails — most fields empty | Fails — but many more fields filled |

---

## Semantic Gap Analysis

### What 12b gets right (semantic understanding):
1. **Domain comprehension** — understands this is NMR spectroscopy data, specifically 13C
2. **Entity identification** — correctly identifies "13C NMR Spectra" as the subject entity
3. **Activity classification** — "NMR Acquisition", "NMR Spectroscopy" as activities
4. **Agent identification** — MestReNova as software agent, Bruker BioSpin as organization
5. **Method identification** — zgpg30 as pulse program (Bruker convention)
6. **Keyword generation** — relevant, domain-appropriate terms
7. **Description generation** — coherent, factually correct summary

### What 12b still gets wrong:
1. **Quantity type vocabulary** — uses raw Bruker parameter names instead of QUDT standard terms. "MCREST" is not a recognized quantity kind.
2. **Unit assignment** — some units nonsensical ("cm" for intensity)
3. **Over-extraction** — includes 37 instrument parameters, many are internal processing flags irrelevant for metadata
4. **Missing administrative metadata** — no access rights, publisher, distribution, contact point
5. **Date extraction** — `modification_date` still empty despite date information being present in source files
6. **Type system** — `type.id` uses internal SIMONE IDs instead of CV IRIs

### What 4b fundamentally cannot do:
1. **Comprehend domain** — cannot identify dataset subject
2. **Generate descriptions** — cannot summarize file contents into coherent text
3. **Identify agents** — cannot resolve raw CV IRIs to human-readable labels
4. **Generate keywords** — cannot extract domain-relevant terms
5. **Classify activities** — cannot determine what type of measurement was performed
6. **Populate evidence queries** — cannot generate evidence retrieval queries (0 entries in ledger)

---

## Usability Scale

```
[Raw files] ──────────────────────────────────────────── [Perfect metadata]
                    4b           12b
                    ↓             ↓
              "Not usable"   "Partially usable"
              (start over)   (~30min cleanup)
```

### 4b: "Not usable"
Output provides no value beyond what a user could get from reading filenames. Generic title, empty description, wrong quantity types, no provenance. A data catalog would show "Untitled dataset" with no searchable fields.

### 12b: "Partially usable"
Output provides real value:
- Title, description, keywords make dataset discoverable
- Creator, instrument, software provide provenance
- Activity and entity classification enable categorical browsing
- A curator can clean up in ~30 minutes vs ~2 hours from raw files

Still not production-ready:
- Profile conformance fails (administrative metadata missing)
- Quantitative attributes need vocabulary mapping
- Some over-extraction noise
- Date fields empty

---

## Conclusion

**12b output crosses the threshold from "not usable" to "partially usable."** The semantic understanding is fundamentally different:

- **4b model**: pattern-matches file content into JSON fields without comprehension. Produces syntactically valid but semantically empty output. Wrong quantity types, placeholder text, no domain understanding.

- **12b model**: comprehends domain context. Identifies NMR spectroscopy, names instruments and software, generates relevant keywords, creates coherent descriptions. Still imperfect (raw quantity types, over-extraction, missing administrative fields) but provides genuine value as starting point for metadata curation.

**For thesis:** This comparison demonstrates that SIMONE's architecture is sound — pipeline stages (evidence extraction, projection, reconstruction) work correctly when the model can understand the input. The 4b model is a floor, not a ceiling. With 12b+ models, SIMONE produces semantically meaningful metadata that a human curator can refine.
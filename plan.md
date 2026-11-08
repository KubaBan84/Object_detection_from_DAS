# Plan implementacji – Detecting Moving Objects in DAS Recordings (wersja na ocenę 5)

> Termin oddania kodu i raportu: **22.11.2026, 23:59**. Prezentacja: **26.11.2026 / 01.12.2026**.
> Plan obejmuje pełny zakres oceny 5: dane rzeczywiste, wykrywanie **grubych i cienkich linii**, prędkości.
> Obejmuje też oba punkty dodatkowe: **wykres v(t)** i **klasteryzację obiektów**.

---

## 0. Wymagania na 5 i gdzie są realizowane

| # | Wymaganie z treści projektu | Etap | Moduł / plik |
|---|---|---|---|
| 1 | Wczytanie **danych rzeczywistych**; 3 ciągłe segmenty po 2 min | 1 | `das/io.py`, `config.SEGMENTS` |
| 2 | Analiza eksploracyjna: statystyki, rozkłady, zmienność w czasie i przestrzeni | 2 | `das/stats.py`, `scripts/run_eda.py` |
| 3 | Analiza częstotliwościowa: **FFT niezależnie dla każdego kanału**, STFT | 3 | `das/spectral.py`, `scripts/run_spectral.py` |
| 4 | Filtracja szumu: pasmo, downsampling, filtry, standaryzacja | 4 | `das/preprocessing.py` |
| 5 | Przetworzenie tak, aby widoczne były **tylko poruszające się obiekty** | 5 | `das/enhancement.py` |
| 6 | Wykrycie linii **grubych (tramwaj/ciężarówka)** | 6A | `das/detection.py` |
| 7 | Wykrycie linii **cienkich, słabych (samochody)** – **warunek oceny 5** | 6B | `das/detection.py` |
| 8 | Grupowanie kandydatów i uśrednianie linii w grupie | 7 | `das/grouping.py` |
| 9 | Prędkość każdego obiektu (km/h, kierunek) | 8 | `das/velocity.py` |
| 10 | (Extra 1) wykres **zmiany prędkości w czasie** | 9 | `das/tracking.py` |
| 11 | (Extra 2) **klasteryzacja** obiektów (gruba/cienka, stała/zmienna prędkość, kształt) | 10 | `das/features.py`, `das/clustering.py` |
| 12 | Ocena liczby i dokładności wykrytych obiektów | 11 | `das/evaluation.py` |
| 13 | Schemat blokowy algorytmu z opisem kroków | 14 | `docs/flowchart.md` → `report/figures/flowchart.png` |
| 14 | Wizualizacje „przed/po” dla każdego kroku, także usuniętego szumu | 4–6, 14 | `das/visualization.py` |
| 15 | Wyniki szczegółowe dla 3 segmentów 2-min | 12 | `scripts/run_pipeline.py` |
| 16 | Opis eksperymentów (udanych i nieudanych), błędów i ograniczeń | 13 | `docs/experiments_log.md` |
| 17 | Źródła danych, kodu i inspiracji | 14 | `docs/references.md` |
| 18 | Kod bez błędów i ostrzeżeń, wektoryzacja, czytelność | całość | `tests/`, `pyproject.toml` (`filterwarnings=error`) |

**Ograniczenia technologiczne.** Algorytm może używać tylko **NumPy, OpenCV, scikit-image, Matplotlib i Pillow**.
Nie wolno używać sieci neuronowych.

- `scipy` **nie jest** na liście dozwolonych bibliotek. Filtry pasmowe, STFT i Welcha piszemy w `numpy.fft`.
  Alternatywa: zapytać prowadzącego mailem o zgodę na `scipy.signal` i zapisać odpowiedź w raporcie.
- `pandas` i `sklearn` też nie są na liście. Tabele zapisujemy modułem `csv`/`json`.
  K-means robimy przez `cv2.kmeans`, a metryki klasteryzacji liczymy w NumPy.
- Jupyter, nbconvert i pytest to tylko narzędzia, nie część algorytmu.

---

## 1. Fakty o danych (sprawdzone na plikach w `data/`)

| | Dane rzeczywiste | Dane syntetyczne |
|---|---|---|
| Archiwum | `data/20240507.rar` (259 plików) | `data/Synthetic Dataset.zip` (6 próbek × 3 pliki) |
| Plik | `HHMMSS.npy`, kształt **(6250, 52)**, float32 | `{id}_signal.npy` **(75000, 59)**, float32 + `{id}_ground_truth.npy` int32 + `{id}_metadata.json` |
| Układ | oś 0 = czas, oś 1 = przestrzeń | jak obok (`matrix_layout: "time_index, space_index"`) |
| dt / fs | 0.0016 s / **625 Hz** (Nyquist 312.5 Hz) | to samo (`axes.dt`) |
| dx | 5.1065 m → 52 kanały ≈ 260 m | to samo (`axes.dx`) |
| Wartości | rząd 1e-6 (std ≈ 1e-6, ekstrema ≈ ±3e-5) | – |
| Ciągłe odcinki | **090017–090227** (14 plików, 140 s) i **090252–094332** (245 plików, 2450 s) | każda próbka = 2 min |
| GT | brak | maska z markerami pojazdów + `type` (car/tram/van…), `direction`, `lane_number` |

Wnioski projektowe:

- Segment 2-minutowy to **12 kolejnych plików** = 75000 × 52 próbek.
  Wszystkie 3 segmenty muszą leżeć w całości w jednym ciągłym odcinku. `io.load_segment` sprawdza to i rzuca wyjątek.
- Przy prędkościach pojazdów 10–150 km/h (2.8–42 m/s) przejazd przez jeden kanał trwa 0.12–1.8 s.
  Samochód jadący 50 km/h przejeżdża cały odcinek kabla w około 19 s.
  Pełna rozdzielczość czasowa 1.6 ms jest więc zbędna dla obrazu trajektorii.
  Decymacja ×25 daje 0.04 s na wiersz, czyli 3000 wierszy na 2 minuty.
- Obraz ma tylko 52 kolumny, więc linie są prawie pionowe. Dlatego interpolujemy oś przestrzeni ×8 (416 kolumn, 0.64 m/px).
  Wtedy 50 km/h daje kąt ok. 40–45°, co znacznie poprawia rozdzielczość kątową Hougha i filtrów kierunkowych.
- Konwencja obrazu jak na rysunku z treści: **czas w pionie (w dół), przestrzeń w poziomie**.
  Linia o prędkości v ma nachylenie Δkol/Δwiersz = v·row_dt/col_dx.

---

## 2. Konwencje kodu (obowiązują na każdym etapie)

- Cała logika siedzi w `src/das/`. Skrypty w `scripts/` tylko parsują argumenty, wołają bibliotekę i zapisują wyniki.
- **Każdy parametr** trafia do dataclass w `config.py`. Wyniki zapisujemy razem z parametrami (JSON), żeby każdy eksperyment był odtwarzalny.
- **Zakaz pętli po pikselach.** Używamy operacji wektorowych NumPy, `cv2.filter2D`, `np.fft`, `sliding_window_view` oraz `np.add.reduceat`/`reshape` do decymacji.
  Pętle są dozwolone tylko po obiektach, kernelach i plikach.
- Funkcje przetwarzające zwracają **słownik etapów pośrednich** (`stages`), bo raport wymaga wizualizacji „przed/po”.
- Type hints, docstringi, komentarze po angielsku. Plan i log eksperymentów po polsku.
- Brak ostrzeżeń w runtime:
  - pytest działa z `filterwarnings=error`;
  - skrypty finalnie uruchamiamy z `python -W error`;
  - uważamy na dzielenie przez zero (MAD=0), `float64`→`uint8` w skimage i log z zera (dodajemy eps).
- Ziarno losowości (`seed`) dla RANSAC i k-means trzymamy w configu.
- Środowisko zarządzane przez **uv**: Python 3.12 (`.python-version`), zależności w `pyproject.toml`, wersje zablokowane w `uv.lock` (wersjonowany).
  - `uv sync` tworzy `.venv` z zależnościami, grupą `dev` i pakietem `das` (editable).
  - Uruchamianie: `uv run python scripts/<skrypt>.py`, `uv run pytest`.
  - Nowe zależności: `uv add <pkg>` (tylko dozwolone biblioteki!) lub `uv add --dev <pkg>` dla narzędzi.
  - Po każdej zmianie zależności odśwież `requirements.txt` dla osób bez uv: `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`.

---

## 3. Etapy implementacji

Każdy etap ma: **cel → pliki → zadania → artefakty → kryterium ukończenia (DoD)**.

### Etap 0 – Środowisko i dane

**Pliki:** `requirements.txt`, `pyproject.toml`, `scripts/extract_data.py`, `data/README.md`.

1. ✅ Środowisko uv zainicjalizowane (`uv sync`): numpy, opencv-python, scikit-image, matplotlib, pillow + dev: jupyter, nbconvert, pytest.
2. Zaimplementuj `extract_data.py`:
   - ZIP rozpakuj przez `zipfile` do `data/synthetic/`;
   - RAR rozpakuj przez systemowy `tar.exe` (libarchive w Windows obsługuje RAR, sprawdzone) do `data/real/`;
   - wywołaj go przez `subprocess.run([...], check=True)`.
3. Zweryfikuj wynik: 259 plików `.npy`, każdy (6250, 52); 6 próbek syntetycznych, każda (75000, 59); poprawny JSON metadanych.
4. **Ustal segmenty.** Odczytaj z arkusza Google (link w treści) przydzielony zakres plików i wpisz go do `config.SEGMENTS`.
   - Praca w parze: 2 przydzielone segmenty + 1 własny. Praca indywidualna: 1 przydzielony + 2 własne. Razem zawsze 3.
   - Segmenty własne wybierz z odcinka 090252–094332 po EDA (Etap 2). Nie mogą nachodzić na przydzielone.
   - Wybierz je tak, by różniły się ruchem: jeden z tramwajami/ciężarówkami, jeden z dużą liczbą słabych linii samochodów.

**DoD:** `uv run python scripts/extract_data.py` kończy się komunikatem OK; `config.SEGMENTS` ma 3 wpisy.

### Etap 1 – Wczytywanie danych

**Pliki:** `src/das/io.py`, `tests/test_io.py`.

1. `list_real_files`, `parse_timestamp`: nazwa `HHMMSS` + data 2024-05-07 → `datetime`.
2. `find_continuous_runs`: sąsiednie pliki muszą różnić się o 10 s. Funkcja ma zwrócić dwa odcinki opisane w sekcji 1.
3. `load_segment(start, n_files=12)`:
   - `np.concatenate` po osi czasu;
   - zwraca macierz (75000, 52) i czas startu;
   - jeśli w zakresie jest przerwa, rzuca `ValueError`;
   - ma opcję `pad_files=1`, która wczytuje po jednym pliku marginesu z każdej strony (jeśli istnieje), żeby filtracja FFT nie psuła brzegów segmentu. Margines przycinamy po przetworzeniu.
4. `load_synthetic(id)`: zwraca (signal, gt, meta); dt i dx bierzemy z `meta["axes"]`.
5. Testy: ciągłość na sztucznej liście nazw, kształt sklejonego segmentu, wyjątek przy przerwie.

**DoD:** testy zielone; `load_segment` dla każdego segmentu z configu zwraca (75000, 52).

### Etap 2 – Analiza eksploracyjna (EDA)

**Pliki:** `src/das/stats.py`, `scripts/run_eda.py`, `src/das/visualization.py` (`plot_waterfall`).

1. **Statystyki globalne po wszystkich 259 plikach**, liczone strumieniowo (`RunningStats`, Welford), bez ładowania całości do RAM:
   - średnia, wariancja i RMS per kanał;
   - globalny histogram w stałych binach (skala log), percentyle, skośność i kurtoza.
   Duża kurtoza oznacza impulsy, więc uzasadnia normalizację odporną (mediana/MAD) zamiast średniej/std.
2. **Statystyki per segment:** `describe`, `per_channel_stats` (RMS, std, MAD na kanał), `per_time_stats` (okno np. 1 s: RMS uśrednione po kanałach).
3. **Zmienność:**
   - mapa RMS (czas × kanał) w oknach 1 s;
   - profil RMS wzdłuż kabla, który pokazuje martwe lub zaszumione kanały, czyli pionowe pasy;
   - przebieg RMS w czasie, który pokazuje zdarzenia globalne, czyli poziome pasy;
   - zmienność RMS kanałów między plikami w ciągu 40 minut.
4. `detect_bad_channels`: odporny z-score RMS kanału (próg ≈ 3.5). Wynik pozwala maskować lub interpolować złe kanały w Etapie 4.
5. Wykresy:
   - waterfall surowego segmentu (|x|, klip do 99 percentyla), wizualnie jak na rysunku w treści;
   - histogramy, boxploty per kanał, mapa RMS, autokorelacja sąsiednich kanałów (spójność przestrzenna).
6. Zapis: `outputs/results/eda_global.json`, `eda_<segment>.json`, rysunki w `outputs/figures/eda/`.

**DoD:** tabela statystyk i co najmniej 6 rysunków gotowych do raportu; lista złych kanałów (może być pusta); wybrane segmenty własne.

### Etap 3 – Analiza częstotliwościowa

**Pliki:** `src/das/spectral.py`, `scripts/run_spectral.py`.

1. `channel_fft`: `np.fft.rfft(x * window, axis=0)`, czyli **niezależnie dla każdego kanału** (wymaganie z treści).
   Wynik to wykres widma amplitudowego: mapa częstotliwość × kanał w skali log oraz widmo uśrednione po kanałach.
2. `welch_psd`: segmentacja przez `sliding_window_view`, okno Hanna, uśrednianie. Daje gładsze widmo do wyboru pasma.
3. `stft`: dla 2–3 kanałów (środkowy oraz kanał z wyraźnym przejazdem tramwaju) policz spektrogram w skali dB.
   Pokaż, jak przejazd pojazdu zmienia treść częstotliwościową: tramwaj ma szerokie pasmo i niskie częstotliwości, samochód słabsze, wyższe pasmo.
4. `band_energy`: energia w pasmach, np. 0–1, 1–5, 5–10, 10–20, 20–50, 50–100, 100–312 Hz.
   Policz ją osobno dla fragmentów „z pojazdem” i „tło”, wybranych automatycznie po RMS: górny i dolny kwantyl okien.
   Stosunek energii pojazd/tło w każdym paśmie to **obiektywne kryterium wyboru `f_low`/`f_high`**.
5. `fk_spectrum`: 2D FFT obrazu (czas × przestrzeń) po decymacji.
   - Na wykres f-k nałóż linie stałej prędkości ±20, ±50, ±100 km/h.
   - Pokaż, że energia pojazdów leży w „wachlarzu” prędkości.
   - Pokaż, że szum skupia się przy k≈0 (zdarzenia jednoczesne na całym kablu, czyli nieskończona prędkość pozorna) i przy f≈0 (stałe pasy kanałów).
6. Sprawdź tonalne zakłócenia (np. 50 Hz i harmoniczne z sieci) i ewentualnie dodaj filtry notch w dziedzinie FFT.

**DoD:**
- wybrane i uzasadnione liczbowo pasmo (wpisane do `PreprocessingParams`);
- rysunki FFT/PSD/STFT/f-k w raporcie;
- notatka w `experiments_log.md`.

### Etap 4 – Usuwanie szumu i budowa obrazu energii

**Pliki:** `src/das/preprocessing.py`, `tests/test_preprocessing.py`.

Kolejność kroków (każdy zapisywany w `stages`):

1. `detrend_channels`: usuń średnią i trend liniowy każdego kanału.
2. `remove_common_mode`: odejmij **medianę po kanałach** w każdej próbce czasu. Usuwa poziome pasy, czyli zakłócenia wspólne dla całego kabla.
   Zapisz też usunięty składnik do wizualizacji „co usunięto”.
3. `bandpass_fft`:
   - `rfft` po osi czasu, maska pasmowa z kosinusowymi zboczami (`taper_hz`), `irfft`;
   - filtr jest zero-fazowy, więc nie przesuwa trajektorii w czasie;
   - opcjonalnie dodaj notch z Etapu 3.
4. `envelope_downsample`: RMS w blokach po 25 próbek (`x.reshape(-1, 25, n_ch)`, potem średnia kwadratów i pierwiastek).
   Daje jednocześnie obwiednię energii i decymację: 75000 → 3000 wierszy, 0.04 s/wiersz.
   Porównaj z wariantem średnia |x| i z prostą decymacją bez obwiedni (eksperyment do raportu).
5. Złe kanały z Etapu 2 interpoluj liniowo z sąsiadów.
6. `robust_normalize`: (x − mediana)/(1.4826·MAD) per kanał. Wyrównuje czułość kanałów, czyli usuwa pionowe pasy.
7. `log_compress_and_clip`: `log1p` z dodatniej części, potem klip do percentyla 99.5.
   Zmniejsza dominację tramwajów, przez co słabe linie samochodów stają się widoczne. **To kluczowe dla oceny 5.**
8. Wygładzenie anizotropowe: lekki `cv2.GaussianBlur` (σ_t > σ_x) lub `cv2.bilateralFilter`. Porównaj z medianowym.
9. `upsample_space`: `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` ×8 w osi przestrzeni.
10. Opcjonalny wariant równoległy: `skimage.exposure.equalize_adapthist` (CLAHE). Porównaj kontrast słabych linii przed i po.

Testy:

- sinus w paśmie przechodzi z błędem < 1%, poza pasmem jest tłumiony > 40 dB;
- common-mode usuwa sygnał identyczny na wszystkich kanałach;
- kształty po decymacji są poprawne.

**DoD:** obraz energii 3000 × 416, na którym gołym okiem widać grube i cienkie linie; rysunek z pełnym łańcuchem etapów i składnikami usuniętymi.

### Etap 5 – Wzmocnienie ukośnych linii (tylko poruszające się obiekty)

**Pliki:** `src/das/enhancement.py`, `tests/test_enhancement.py`.

1. **Filtr wachlarzowy f-k (`fk_velocity_filter`):**
   - Pracuj na obrazie po kroku 7 (przed upsamplingiem, czyli z rzeczywistym dx), po odjęciu mediany.
   - Pad przestrzenny 52 → 128 z oknem Tukeya, żeby ograniczyć przecieki na brzegach.
   - Maska przepuszcza tylko |v| = |f/k| ∈ [v_min, v_max] = [10, 150] km/h, ze stożkowymi przejściami (`fk_taper`).
   - **Osobna maska dla każdego kierunku**: znak f·k wybiera ruch w stronę rosnących lub malejących pozycji.
     Powstają dwa obrazy (+1 i −1), co rozdziela **krzyżujące się linie** (widoczne na rysunku z treści) i upraszcza detekcję.
   - Fizyczne uzasadnienie:
     - pozioma linia ma v→∞; to zdarzenie jednoczesne lub szybka fala sejsmiczna o prędkościach setek m/s;
     - pionowa linia ma v→0; to stały szum kanału;
     - obie leżą poza wachlarzem i zostają usunięte.
   - Zapisz obraz usuniętego składnika (wejście − wyjście).
2. `subtract_background`: biały top-hat (`cv2.morphologyEx(MORPH_TOPHAT)`) lub odjęcie mediany ruchomej. Usuwa wolnozmienne tło.
3. **Bank filtrów kierunkowych (`oriented_line_kernels` + `directional_filter_bank`):**
   - Kernele mają postać odcinka o długości L (np. 41 px) i gaussowskim profilu poprzecznym, ze średnią zero i normą jednostkową, czyli filtr dopasowany.
   - Kąty bierzemy tylko z przedziału odpowiadającego [v_min, v_max] dla danego kierunku (`velocity_to_angle`), np. 24 kąty.
   - Splot przez `cv2.filter2D`. Wynik to maksimum odpowiedzi i indeks kąta (lokalna orientacja i prędkość).
   - Uzasadnienie: całkowanie wzdłuż linii zwiększa SNR o rząd √L. **To główny mechanizm wydobycia cienkich, słabych linii.**
4. Ścieżka alternatywna do porównania w raporcie:
   - `ridge_filter`, czyli `skimage.filters.sato`/`meijering` (detektor grzbietów z hesjanu);
   - `skimage.filters.gabor` jako alternatywa dla kerneli liniowych.
5. `enhance()` zwraca słownik: `fk_pos`, `fk_neg`, `removed_fk`, `bg_removed_*`, `response_*`, `angle_*`.

Test: syntetyczny obraz zawiera linię 50 km/h, linię poziomą, linię pionową i szum. Po filtrze zostaje tylko linia ukośna, we właściwym kanale kierunku.

**DoD:**
- na rzeczywistym segmencie obraz odpowiedzi ma wyraźne linie cienkie i grube, a pasy poziome i pionowe są usunięte;
- rysunek „przed/po/usunięte” dla f-k i filtra kierunkowego.

### Etap 6 – Detekcja linii: dwie ścieżki przetwarzania

**Pliki:** `src/das/detection.py`, `tests/test_detection.py`.

Każdą ścieżkę uruchamiamy osobno dla kierunku +1 i −1.

**6A. Linie grube i silne (`detect_strong_lines`)**

1. Progowanie Otsu (`skimage.filters.threshold_otsu`) na obrazie odpowiedzi. Alternatywnie próg k·σ_robust (`robust_noise_sigma`).
2. Otwarcie morfologiczne elementem liniowym zorientowanym w dominującym kierunku (kilka kątów, potem max).
   Usuwa plamki i podtrzymuje struktury wydłużone.
3. `skimage.measure.label` + `regionprops`. Odrzuć obiekty o małym polu, małej ekscentryczności, złej orientacji lub zbyt krótkim zasięgu przestrzennym (< `min_channels`).
4. Dla każdego komponentu dopasuj prostą przez RANSAC (`skimage.measure.ransac` + `LineModelND`) na pikselach ważonych odpowiedzią.
   Dodatkowo policz grubość jako pole/długość.
5. Zwróć segmenty `LineSegment(source="strong")` i **maskę obiektów silnych** (do ścieżki 6B).

**6B. Linie cienkie i słabe (`detect_weak_lines`) – warunek oceny 5**

1. Zamaskuj obiekty silne. Maskę dilatujemy o `strong_mask_dilation_px`, a w zamaskowanym obszarze odpowiedź ustawiamy na 0.
   Dzięki temu „halo” tramwaju nie generuje fałszywych słabych linii, a próg dla słabych można obniżyć.
2. Progi wyznaczamy z danych: σ = 1.4826·MAD odpowiedzi w tle.
   `skimage.filters.apply_hysteresis_threshold(low=2σ, high=4σ)` podąża za słabym, ale ciągłym grzbietem.
3. `skimage.morphology.remove_small_objects`, następnie `skeletonize`.
4. `cv2.HoughLinesP` na szkielecie (`threshold`, `minLineLength`, `maxLineGap` z configu).
   Odrzuć segmenty o kącie spoza zakresu prędkości lub o niewłaściwym kierunku.
5. Komplementarnie `hough_global`, czyli `skimage.transform.hough_line` + `hough_line_peaks` z `min_distance`/`min_angle`.
   Sprawdza się dla długich, prostych i bardzo słabych linii, bo akumuluje głosy wzdłuż całej linii.
   Porównaj obie metody w raporcie (recall cienkich linii na danych syntetycznych).
6. Opcjonalnie dla wydajności i lepszej lokalności: przetwarzaj w oknach czasowych 30 s z zakładką 50%, a segmenty scalaj w Etapie 7.

Test: obraz z jedną grubą i jedną cienką linią (SNR ≈ 1.5) + szum. Obie linie mają zostać wykryte, cienka przez ścieżkę 6B.

**DoD:**
- na danych syntetycznych recall dla `car` ≥ ~0.8, a dla `tram` ≈ 1.0 (cele orientacyjne, raportujemy rzeczywiste);
- na segmentach rzeczywistych wizualnie wykryta większość cienkich linii.

### Etap 7 – Grupowanie kandydatów i uśrednianie linii

**Pliki:** `src/das/grouping.py`.

1. Każdy segment opisz w przestrzeni fizycznej parą **(v [km/h], t_ref [s])**, gdzie t_ref to czas przejścia przez środek kabla (`segment_to_params`).
   Ta parametryzacja nie zależy od długości segmentu.
2. `group_segments`:
   - macierz odległości po parach (wektorowo), łączenie gdy |Δv| < `tol_velocity_kmh` (lub względnie < 15%) i |Δt_ref| < `tol_time_s`;
   - grupy wyznaczamy jako składowe spójne (union-find albo `skimage.measure.label` na macierzy sąsiedztwa).
   Scala to:
   - podwójne krawędzie grubej linii;
   - pocięte fragmenty cienkiej linii;
   - duplikaty z okien czasowych;
   - wyniki `HoughLinesP` i `hough_line`.
3. `merge_group`: średnia ważona (waga = długość × średnia odpowiedź) albo ponowne dopasowanie prostej do wszystkich końców i pikseli grupy.
   Wynikiem jest jeden `DetectedObject` z zasięgiem (x_start..x_end, t_start..t_end).
4. `remove_duplicates`: jeśli obiekt „weak” pokrywa się z „strong”, zostaje strong.
   Odrzuć obiekty z zasięgiem < `min_channels` kanałów lub prędkością poza 5–150 km/h (filtr wiarygodności).
5. Etykieta `kind` (strong/weak): według ścieżki pochodzenia, skorygowana grubością i energią (Etap 10).

**DoD:** jedna linia na jeden pojazd; brak wielokrotnych linii na tramwaju; rysunek „kandydaci → grupy → linia średnia”.

### Etap 8 – Wyznaczenie prędkości

**Pliki:** `src/das/velocity.py`, `tests/test_velocity.py`.

1. `slope_to_velocity`: v = (Δkol·col_dx)/(Δwiersz·row_dt) [m/s], potem ×3.6 → km/h.
   Znak oznacza kierunek (+ w stronę rosnącej pozycji na kablu).
2. `fit_constant_velocity`: na punktach z trackingu (Etap 9) dopasuj prostą metodą najmniejszych kwadratów i podaj R².
   To dokładniejsza prędkość niż z samego Hougha i ją podajemy w tabelach.
3. Wizualizacja jak na rysunku z treści: czerwone linie na waterfallu, etykieta „xx.xx km/h”.
   Linie silne rysujemy ciągłą kreską, słabe przerywaną. Oś czasu w formacie HH:MM:SS, oś przestrzeni w metrach.
4. Tabela obiektów per segment (CSV): id, kind, kierunek, v [km/h], t_start, t_end, x_start, x_end, score, R².

Test: linia o znanym nachyleniu daje znaną prędkość; zgodność konwencji znaku z `direction` z metadanych syntetycznych.

**DoD:** dla danych syntetycznych MAE prędkości raportowane per typ pojazdu; dla danych rzeczywistych tabele i rysunki.

### Etap 9 – (Extra 1) Śledzenie i prędkość w funkcji czasu

**Pliki:** `src/das/tracking.py`, `visualization.plot_velocity_profiles`.

1. `track_ridge`:
   - dla każdego obiektu i każdego kanału w zasięgu wyznacz przewidywany czas z linii średniej;
   - w oknie ±`search_halfwidth_s` znajdź maksimum energii (argmax wektorowo na wyciętym pasie);
   - doprecyzuj położenie interpolacją paraboliczną (subpróbkowo).
   Wynik to punkty (x_m, t_s).
2. `fit_trajectory_ransac`: odrzuć punkty odstające, np. przecięcie z innym pojazdem.
3. `velocity_profile`: lokalne nachylenie dx/dt w oknie przesuwnym `velocity_window_channels` (wygładzony gradient lub lokalna regresja liniowa).
   Wynik to v(t) w km/h.
4. `speed_variability`:
   - porównanie residuów dopasowania liniowego i kwadratowego (x(t) = a t² + b t + c);
   - odchylenie std v(t) i średnie przyspieszenie.
   Służy do etykiety „stała / zmienna prędkość”.
5. Wykres v(t): osobne panele dla kilku obiektów oraz zbiorczy wykres dla segmentu z kolorem wg klastra.
   Trajektorie nieliniowe (krzywe) nanieś na waterfall.

**DoD:** wykres v(t) dla każdego obiektu z każdego segmentu; zidentyfikowane przypadki hamowania i przyspieszania (np. przy skrzyżowaniu lub przystanku).

### Etap 10 – (Extra 2) Cechy i klasteryzacja obiektów

**Pliki:** `src/das/features.py`, `src/das/clustering.py`, `scripts/run_clustering.py`.

1. Cechy obiektu (`object_features`):
   - **geometria:** |v|, kierunek, std v(t), współczynnik krzywizny (a z fitu kwadratowego), R², długość czasu, zasięg przestrzenny;
   - **grubość linii** (`line_thickness`): FWHM profilu poprzecznego do trajektorii [s], uśredniona po kanałach;
   - **energia:** maksimum i średnia odpowiedzi wzdłuż trajektorii, energia całkowita;
   - **widmo:** centroid widmowy i stosunek energii pasmo niskie/wysokie surowego sygnału wzdłuż trajektorii (okno wokół t(x) dla każdego kanału);
   - **kształt:** liczba „podlinii” (np. tramwaj = kilka wózków lub osi, czyli równoległe prążki); profil autokorelacji poprzecznej.
2. `standardize` (z-score), `choose_k` dla k = 2..6 według silhouette (`silhouette_score` w NumPy) i metodą łokcia (compactness z `cv2.kmeans`).
3. `kmeans` przez `cv2.kmeans` (`KMEANS_PP_CENTERS`, `attempts=10`).
4. Interpretacja:
   - centroidy klastrów w jednostkach fizycznych;
   - nazwy klastrów, np. „gruba/szybka/stała”, „cienka/stała”, „zmienna prędkość”;
   - galeria wycinków obrazu (miniatury trajektorii) dla każdego klastra, czyli „kształt linii”.
5. `pca_2d` (SVD) daje scatter 2D z kolorem klastra.
   Dodatkowo pokaż reguły proste (progi grubość/energia) jako baseline dla porównania.
6. Walidacja na danych syntetycznych: `cluster_purity` względem `type` (car/tram/van).
   Na rzeczywistych: zgodność z etykietami ręcznymi thick/thin (Etap 11).
7. Klasteryzację liczymy na obiektach ze **wszystkich 3 segmentów razem** (więcej próbek). Można też dodać inne segmenty z ciągłego odcinka, co treść dopuszcza.

**DoD:** wykres PCA, tabela centroidów, galerie, opis klastrów w raporcie.

### Etap 11 – Ewaluacja ilościowa

**Pliki:** `src/das/evaluation.py`, `scripts/evaluate_synthetic.py`, `scripts/annotate_real.py`, `data/annotations/manual_labels.json`.

1. **Dane syntetyczne:**
   - `gt_objects_from_mask`: dla każdego `marker_value` weź punkty (x, t) z maski, dopasuj prostą i wyznacz v_true, t_ref, typ i kierunek;
   - uruchom pipeline (dt i dx z metadanych; 59 kanałów, więc kod nie może zakładać 52);
   - `match_detections`: zachłanne parowanie 1-1 po koszcie (|Δt_ref|/tol_t + |Δv|/tol_v) z bramkowaniem;
   - `detection_metrics`: TP, FP, FN, precision, recall, F1 **osobno dla car (cienkie) i tram (grube)**, bo to dowód realizacji oceny 5;
   - `velocity_errors`: MAE, RMSE, MAPE;
   - `trajectory_iou` między maską wykrytych trajektorii a maską GT (z tolerancją).
2. **Dane rzeczywiste (brak GT):**
   - `annotate_real.py`: interaktywnie (`plt.ginput`) zaznacz początek i koniec każdej widocznej linii w 3 segmentach, z etykietą thick/thin;
   - zapisz do `manual_labels.json` (wersjonowane);
   - te same metryki co wyżej liczone względem adnotacji. Zaznacz w raporcie, że to ocena subiektywna (ograniczenie metody).
3. Wyniki zapisz do `outputs/results/eval_*.json` i tabel w raporcie.

**DoD:** tabela metryk dla 6 próbek syntetycznych i 3 segmentów rzeczywistych, w podziale na thick i thin.

### Etap 12 – Pipeline end-to-end i wyniki dla 3 segmentów

**Pliki:** `src/das/pipeline.py`, `scripts/run_pipeline.py`.

1. `run_pipeline` łączy Etapy 4→5→6A/6B→7→8→9→10 (cechy) i mierzy czas każdego etapu `time.perf_counter`.
2. `save_result`:
   - CSV obiektów i JSON z parametrami oraz czasami;
   - opcjonalnie cache etapów w `.npy`, żeby notebook raportu nie przeliczał wszystkiego.
3. `run_pipeline.py --segment all` dla każdego segmentu z configu produkuje:
   - waterfall surowy, obraz po preprocessingu, obraz po f-k (oba kierunki), odpowiedź filtra;
   - kandydatów, wynik końcowy z prędkościami, wykres v(t), tabelę obiektów;
   - liczbę obiektów strong/weak per kierunek.
4. Ten sam skrypt działa dla `--synthetic <id>` i dowolnego `--start HHMMSS`, co pokazuje, że podejście jest ogólne i nic nie jest zakodowane na sztywno.
5. Cel wydajności: segment 2-min w ≲ 20–30 s na laptopie. Raportujemy tabelę czasów etapów.

**DoD:** jedna komenda generuje komplet wyników dla wszystkich 3 segmentów bez błędów i ostrzeżeń (`uv run python -W error scripts/run_pipeline.py --segment all`).

### Etap 13 – Strojenie parametrów, ablacja, eksperymenty

**Pliki:** `scripts/tune_parameters.py`, `docs/experiments_log.md`.

1. Siatka po kluczowych parametrach (`f_low`/`f_high`, decymacja, progi histerezy, długość kernela, `tol_*`).
   Optymalizujemy F1 (car + tram) na syntetycznych i F1 na adnotacjach rzeczywistych.
   **Nie stroimy pod pojedyncze linie**, bo treść zabrania hardkodowania.
2. **Ablacja:** wyłączamy kolejno common-mode, f-k, log-kompresję, bank filtrów i maskowanie silnych linii.
   Wynikiem jest tabela zmiany recall cienkich linii, czyli dowód, który krok za co odpowiada.
3. **Porównanie metod:**
   - HoughLinesP vs `hough_line` (globalny) vs RANSAC na komponentach;
   - filtr kierunkowy vs Sato/Meijering vs Gabor;
   - RMS vs |x|.
4. Analiza wrażliwości: wykres F1 w funkcji progu histerezy.
5. Każdą próbę (także nieudaną) wpisujemy do `experiments_log.md`: co, parametry, wynik, decyzja.
   Typowe nieudane próby do opisania:
   - Canny + Hough na surowym obrazie (dużo fałszywych linii od pasów);
   - pojedynczy próg globalny (gubi cienkie linie);
   - f-k bez paddingu (artefakty brzegowe).

**DoD:** tabela ablacji, porównanie metod i finalny zestaw parametrów zapisany jako domyślny w `config.py`.

### Etap 14 – Raport i schemat blokowy

**Pliki:** `report/report.ipynb` (szkielet sekcji już utworzony), `scripts/make_report_figures.py`, `docs/flowchart.md`, `docs/references.md`.

1. Forma oddania: **`report.ipynb` + `report.html`**, przez `uv run jupyter nbconvert --to html --execute report/report.ipynb`.
   Opcjonalnie PDF przez drukowanie HTML do PDF. Kod źródłowy `.py` z `src/` i `scripts/` dołączamy w całości.
2. Schemat blokowy: diagram mermaid z `docs/flowchart.md` wyeksportuj do PNG (mermaid.live lub draw.io) jako `report/figures/flowchart.png`.
   Do każdego bloku dodaj opis: co robi i po co.
3. `make_report_figures.py` regeneruje wszystkie rysunki w zmniejszonej rozdzielczości (dpi ≈ 100–110, waterfall przeskalowany `cv2.INTER_AREA`), żeby raport nie był za duży.
4. Treść raportu (sekcje są już w notebooku):
   1. dane;
   2. EDA;
   3. analiza częstotliwościowa;
   4. schemat;
   5. preprocessing z obrazami „przed/po/usunięte”;
   6. wzmocnienie;
   7. detekcja strong/weak i grupowanie;
   8. prędkości;
   9. v(t);
   10. klasteryzacja;
   11. **wyniki szczegółowe dla 3 segmentów**;
   12. ewaluacja, ablacja i eksperymenty nieudane;
   13. wnioski, błędy merytoryczne i metodyczne oraz ograniczenia;
   14. źródła.
5. Ograniczenia do omówienia:
   - brak GT dla danych rzeczywistych;
   - pojazdy jadące równolegle lub w kolumnie zlewają się w jedną linię;
   - przecięcia linii;
   - nieznana dokładna geometria kabla (kabel może nie być równoległy do jezdni na całej długości, co daje błąd prędkości);
   - krótki odcinek kabla (≈260 m) ogranicza dokładność v(t);
   - różnice między danymi syntetycznymi a rzeczywistymi.
6. Źródła: dane, notebook przykładowy z kursu, dokumentacja bibliotek, literatura (lista w `docs/references.md`; przed cytowaniem zweryfikować szczegóły bibliograficzne).
   Podajemy też narzędzia AI, jeśli były użyte, z opisem do czego.

**DoD:** notebook wykonuje się od początku do końca bez błędów; HTML < ~20 MB; wszystkie punkty z sekcji „Report” treści pokryte.

### Etap 15 – Kontrola jakości i oddanie

1. `uv run pytest` z `filterwarnings=error`: wszystkie testy zielone (zdjąć `skip` z szablonów testów).
2. Czysty klon repo: `uv sync`, `uv run python scripts/extract_data.py`, `uv run python scripts/run_pipeline.py --segment all`. Wszystko przechodzi bez ręcznych poprawek.
   Sprawdź też ścieżkę bez uv: `pip install -r requirements.txt && pip install -e .`.
3. Przegląd kodu:
   - brak pętli po pikselach, brak martwego kodu;
   - docstringi i komentarze na miejscu;
   - brak hardkodowanych pozycji linii;
   - parametry tylko w `config.py`.
4. Paczka do oddania: `src/`, `scripts/`, `tests/`, `report/report.ipynb`, `report/report.html`, `pyproject.toml`, `uv.lock`, `.python-version`, `requirements.txt`, `README.md`.
   Bez danych, chyba że prowadzący wymaga inaczej.
5. Przygotowanie do prezentacji (26.11 / 01.12): krótkie demo `run_pipeline.py` na jednym segmencie, kluczowe rysunki, umiejętność uzasadnienia każdego kroku i parametru.

---

## 4. Harmonogram (08.11 – 22.11.2026)

| Dzień | Etapy | Rezultat |
|---|---|---|
| 08–09.11 | 0, 1, 2 | dane rozpakowane, loader, EDA, wybrane segmenty |
| 10.11 | 3 | pasmo filtracji uzasadnione |
| 11–12.11 | 4, 5 | obraz energii + f-k + bank filtrów |
| 13–15.11 | 6, 7, 8 | detekcja strong + weak, grupowanie, prędkości (**rdzeń oceny 5**) |
| 16.11 | 11 | metryki na syntetycznych + adnotacje 3 segmentów rzeczywistych |
| 17.11 | 9 | v(t) |
| 18.11 | 10 | klasteryzacja |
| 19.11 | 12, 13 | pełne przebiegi, strojenie, ablacja |
| 20–21.11 | 14 | raport + schemat blokowy |
| 22.11 | 15 | kontrola jakości, **oddanie przed 23:59** |
| 23–25.11 | – | przygotowanie prezentacji |

Bufor: jeśli Etap 6B (cienkie linie) się opóźnia, ma on priorytet nad Etapami 9 i 10, bo to warunek oceny 5, a 9 i 10 to punkty dodatkowe.

---

## 5. Ryzyka i plan B

| Ryzyko | Objaw | Reakcja |
|---|---|---|
| Cienkie linie giną w szumie | niski recall `car` | dłuższe kernele, log-kompresja, maskowanie silnych, `hough_line` globalny, mniejsza decymacja |
| Dużo fałszywych linii | niska precyzja | ostrzejszy filtr f-k, min. zasięg w kanałach, filtr wiarygodności v, wyższy próg `high` histerezy |
| Grube linie rozbite na wiele | wiele obiektów na tramwaju | większe `tol_*` w grupowaniu, zamknięcie morfologiczne w 6A |
| Krzyżujące się linie | zlane obiekty | rozdzielenie kierunków w f-k (już w planie), RANSAC przy trackingu |
| Artefakty brzegowe FFT | fałszywe linie na brzegach segmentu | `pad_files=1`, okno Tukeya, padding przestrzenny |
| Brak zgody na scipy | – | wszystko w `numpy.fft` (już tak zaplanowane) |
| Wydajność | > 1 min na segment | filtr kierunkowy na obrazie zdecymowanym, przetwarzanie oknami, cache etapów |

---

## 6. Struktura repozytorium

```
Object_detection_from_DAS/
├── plan.md                      # ten plan
├── README.md                    # instalacja, uruchomienie, układ katalogów
├── pyproject.toml, uv.lock, .python-version   # uv: zależności, lockfile, wersja Pythona
├── requirements.txt             # eksport z uv (dla instalacji przez pip)
├── .gitignore                   # m.in. .venv/, dane, outputs/
├── Project 1 2026.pdf           # treść zadania
├── data/
│   ├── README.md                # opis danych i ekstrakcji
│   ├── 20240507.rar, Synthetic Dataset.zip   # (ignorowane przez git)
│   ├── real/                    # 259 × HHMMSS.npy        (ignorowane)
│   ├── synthetic/               # {id}_signal/_ground_truth/_metadata (ignorowane)
│   └── annotations/manual_labels.json        # ręczne adnotacje (wersjonowane)
├── src/das/
│   ├── config.py                # stałe, ścieżki, SEGMENTS, dataclassy parametrów   (Etap 1)
│   ├── io.py                    # wczytywanie, ciągłość, segmenty                    (Etap 1)
│   ├── stats.py                 # EDA, statystyki strumieniowe                        (Etap 2)
│   ├── spectral.py              # FFT per kanał, Welch, STFT, f-k                     (Etap 3)
│   ├── preprocessing.py         # filtracja, obwiednia, decymacja, normalizacja       (Etap 4)
│   ├── enhancement.py           # filtr f-k wachlarzowy, bank filtrów kierunkowych    (Etap 5)
│   ├── detection.py             # ścieżka strong (6A) i weak (6B), Hough              (Etap 6)
│   ├── grouping.py              # grupowanie kandydatów, DetectedObject               (Etap 7)
│   ├── velocity.py              # prędkość, konwersje                                 (Etap 8)
│   ├── tracking.py              # śledzenie grzbietu, v(t)                            (Etap 9)
│   ├── features.py              # cechy obiektów                                      (Etap 10)
│   ├── clustering.py            # cv2.kmeans, silhouette, PCA                         (Etap 10)
│   ├── evaluation.py            # metryki vs GT / adnotacje                           (Etap 11)
│   ├── visualization.py         # wszystkie wykresy                                   (cały projekt)
│   └── pipeline.py              # orkiestracja end-to-end                             (Etap 12)
├── scripts/                     # extract_data, run_eda, run_spectral, run_pipeline,
│                                # evaluate_synthetic, annotate_real, tune_parameters,
│                                # run_clustering, make_report_figures
├── tests/                       # testy jednostkowe per moduł
├── notebooks/                   # notatniki eksploracyjne (01_eda, 02_spectral, 03_detection_dev …)
├── report/
│   ├── report.ipynb             # raport końcowy (szkielet sekcji gotowy)
│   └── figures/                 # rysunki do raportu (zmniejszone)
├── docs/
│   ├── flowchart.md             # schemat blokowy (mermaid)
│   ├── experiments_log.md       # dziennik eksperymentów
│   └── references.md            # źródła
└── outputs/                     # wyniki generowane (ignorowane): figures/, results/, cache/
```

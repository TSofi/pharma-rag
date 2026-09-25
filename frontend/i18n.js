// Interface translations. The "Answer in" selector also switches the interface language
// ("Same as my question" keeps the interface in English).
window.I18N = (() => {
  const STR = {
    en: {
      eyebrow: "Ask about your drug",
      h1a: "Know your medicine.", h1b: "Straight from the label.",
      market: "Market", answerIn: "Answer in",
      auto: "Same as my question",
      mkt_us: "United States · FDA", mkt_at: "Austria · EMA + BASG (coming soon)",
      mkt_pl: "Poland · EMA + URPL (coming soon)", mkt_ua: "Ukraine · State Register (coming soon)",
      placeholder: "Ask about any drug…",
      library: "Drug library", libNote: "Official FDA labels indexed in the vector database. Click one to ask about it.",
      filter: "Filter…", libAsk: "What is {drug} used for?",
      answer: "Answer", sources: "Sources", showUncited: "show uncited",
      showFull: "Show full passage", collapse: "Collapse", openLabel: "Open label on DailyMed ↗",
      scoreTitle: "Cosine similarity between your question and this passage",
      searching: "Searching labels and writing a sourced answer…",
      metaFound: "{cited} of {total} retrieved passages cited · {secs}s",
      metaVersions: "label versions: {dates}",
      metaNone: "No passage was relevant enough to answer from.",
      metaPlayful: "Not in the labels, but we appreciate the creativity.",
      metaScore: "you scored {n} while waiting 🎮",
      interpreted: "Interpreted “{typed}” as <b>{matched}</b>.",
      searchedFor: "Searched the English labels for: “{q}”",
      busy: "The AI model is overloaded right now (free tier). Your question is fine, so please try again in a moment.",
      wrong: "Something went wrong.", retry: "Try again", details: "Technical details",
      noApi: "Can't reach the API. Is the backend running?", apiErr: "The API returned an error.",
      tabGame: "Play while you wait", tabFact: "Did you know?",
      gameHint: "Space / ↑ / tap to jump · score {score} · best {best}",
      foot1: "Answers come only from official drug labels. Each sentence links to the exact label section it came from, and if the label doesn't say it, PharmaRAG says so.",
      foot2: "Demo project for educational purposes only. Not medical advice. Always consult the full prescribing information.",
      chips: [
        "Can amlodipine cause ankle swelling?", "Can I take ibuprofen with lisinopril?",
        "Is dizziness a side effect of losartan?", "Can sertraline and tramadol be taken together?",
        "What should I avoid while taking warfarin?", "Can metformin cause stomach problems?",
      ],
    },
    uk: {
      eyebrow: "Запитай про свій препарат",
      h1a: "Знай свої ліки.", h1b: "Просто з інструкції.",
      market: "Ринок", answerIn: "Мова",
      auto: "Як у питанні",
      mkt_us: "США · FDA", mkt_at: "Австрія · EMA + BASG (незабаром)",
      mkt_pl: "Польща · EMA + URPL (незабаром)", mkt_ua: "Україна · Держреєстр (незабаром)",
      placeholder: "Запитай про будь-який препарат…",
      library: "Бібліотека ліків", libNote: "Офіційні інструкції FDA у векторній базі. Натисни на препарат, щоб запитати про нього.",
      filter: "Пошук…", libAsk: "Для чого використовують {drug}?",
      answer: "Відповідь", sources: "Джерела", showUncited: "показати нецитовані",
      showFull: "Показати весь фрагмент", collapse: "Згорнути", openLabel: "Відкрити інструкцію на DailyMed ↗",
      scoreTitle: "Косинусна схожість між питанням і цим фрагментом",
      searching: "Шукаю в інструкціях і пишу відповідь із джерелами…",
      metaFound: "процитовано {cited} з {total} знайдених фрагментів · {secs} с",
      metaVersions: "версії інструкцій: {dates}",
      metaNone: "Жоден фрагмент не був достатньо релевантним для відповіді.",
      metaPlayful: "В інструкціях такого немає, але креативність оцінили.",
      metaScore: "твій рахунок у грі: {n} 🎮",
      interpreted: "«{typed}» розпізнано як <b>{matched}</b>.",
      searchedFor: "Шукали в англомовних інструкціях: «{q}»",
      busy: "Модель ШІ зараз перевантажена (безкоштовний тариф). З питанням усе гаразд, спробуй ще раз за хвилинку.",
      wrong: "Щось пішло не так.", retry: "Спробувати ще раз", details: "Технічні деталі",
      noApi: "Немає зв'язку з сервером. Бекенд запущений?", apiErr: "Сервер повернув помилку.",
      tabGame: "Пограй, поки чекаєш", tabFact: "А ти знав(ла)?",
      gameHint: "Пробіл / ↑ / тап — стрибок · рахунок {score} · рекорд {best}",
      foot1: "Відповіді беруться лише з офіційних інструкцій. Кожне речення посилається на конкретний розділ інструкції, а якщо там цього немає, PharmaRAG так і скаже.",
      foot2: "Демонстраційний навчальний проєкт. Не є медичною порадою. Завжди звіряйтеся з повною інструкцією.",
      chips: [
        "Чи може амлодипін викликати набряк щиколоток?", "Чи можна приймати ібупрофен разом із лізиноприлом?",
        "Чи є запаморочення побічною дією лозартану?", "Чи можна поєднувати сертралін і трамадол?",
        "Чого уникати під час прийому варфарину?", "Чи може метформін викликати проблеми зі шлунком?",
      ],
    },
    pl: {
      eyebrow: "Zapytaj o swój lek",
      h1a: "Poznaj swój lek.", h1b: "Prosto z ulotki.",
      market: "Rynek", answerIn: "Język",
      auto: "Jak w pytaniu",
      mkt_us: "USA · FDA", mkt_at: "Austria · EMA + BASG (wkrótce)",
      mkt_pl: "Polska · EMA + URPL (wkrótce)", mkt_ua: "Ukraina · Rejestr Państwowy (wkrótce)",
      placeholder: "Zapytaj o dowolny lek…",
      library: "Biblioteka leków", libNote: "Oficjalne ulotki FDA w bazie wektorowej. Kliknij lek, aby o niego zapytać.",
      filter: "Szukaj…", libAsk: "Do czego służy {drug}?",
      answer: "Odpowiedź", sources: "Źródła", showUncited: "pokaż niecytowane",
      showFull: "Pokaż cały fragment", collapse: "Zwiń", openLabel: "Otwórz ulotkę na DailyMed ↗",
      scoreTitle: "Podobieństwo kosinusowe między pytaniem a tym fragmentem",
      searching: "Przeszukuję ulotki i piszę odpowiedź ze źródłami…",
      metaFound: "zacytowano {cited} z {total} znalezionych fragmentów · {secs} s",
      metaVersions: "wersje ulotek: {dates}",
      metaNone: "Żaden fragment nie był wystarczająco trafny, by odpowiedzieć.",
      metaPlayful: "Tego nie ma w ulotkach, ale doceniamy kreatywność.",
      metaScore: "twój wynik w grze: {n} 🎮",
      interpreted: "„{typed}” rozpoznano jako <b>{matched}</b>.",
      searchedFor: "Przeszukano angielskie ulotki pod kątem: „{q}”",
      busy: "Model AI jest teraz przeciążony (darmowy plan). Z pytaniem wszystko w porządku, spróbuj ponownie za chwilę.",
      wrong: "Coś poszło nie tak.", retry: "Spróbuj ponownie", details: "Szczegóły techniczne",
      noApi: "Brak połączenia z serwerem. Czy backend działa?", apiErr: "Serwer zwrócił błąd.",
      tabGame: "Zagraj, czekając", tabFact: "Czy wiesz, że…?",
      gameHint: "Spacja / ↑ / dotknij — skok · wynik {score} · rekord {best}",
      foot1: "Odpowiedzi pochodzą wyłącznie z oficjalnych ulotek leków. Każde zdanie odsyła do konkretnej sekcji ulotki, a jeśli ulotka tego nie mówi, PharmaRAG to przyzna.",
      foot2: "Projekt demonstracyjny w celach edukacyjnych. To nie jest porada medyczna. Zawsze sprawdzaj pełną ulotkę.",
      chips: [
        "Czy amlodypina może powodować obrzęk kostek?", "Czy mogę brać ibuprofen z lizynoprylem?",
        "Czy zawroty głowy to skutek uboczny losartanu?", "Czy można łączyć sertralinę z tramadolem?",
        "Czego unikać podczas przyjmowania warfaryny?", "Czy metformina może powodować problemy z żołądkiem?",
      ],
    },
  };

  let lang = "en";
  const t = (key, vars = {}) => {
    const s = (STR[lang] && STR[lang][key]) ?? STR.en[key] ?? key;
    return typeof s === "string" ? s.replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? "") : s;
  };

  // Elements with data-i18n="key" get their text; data-i18n-ph="key" sets a placeholder.
  function apply(newLang) {
    lang = STR[newLang] ? newLang : "en";
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
    // Example questions: localized ones plus one in each other supported language as a teaser.
    const chips = document.getElementById("chips");
    if (chips) {
      const others = { en: ["Для чого використовують метформін?", "Czy mogę brać ibuprofen z Eliquisem?"],
                       uk: ["Can I take ibuprofen with lisinopril?", "Czy mogę brać ibuprofen z Eliquisem?"],
                       pl: ["Can I take ibuprofen with lisinopril?", "Чи можуть від аторвастатину боліти м'язи?"] }[lang];
      chips.innerHTML = [...t("chips"), ...others].map((c) => `<button type="button">${c}</button>`).join("");
    }
  }

  return { t, apply, get lang() { return lang; } };
})();

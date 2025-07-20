setInterval(() => {
    if (!document.hasFocus()) return;
    const urlParams = new URLSearchParams(window.location.search);

    fetch("_static/rand")
        .then((r) => r.text())
        .then((v) => {
            v = v.trim();
            const rand = urlParams.get("rand");
            if (rand === val) return;

            window.location.href = `?rand=${val}`;
            if (rand && rand != val) window.reload();
        });
}, 1000);

// const languages = ["python"];
// const scripts = [
//     "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js",
// ];

// const link = document.createElement("link");
// link.href =
//     "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/github-dark.min.css";
// link.rel = "stylesheet";
// document.head.appendChild(link);

// for (const src of scripts) {
//     const script = document.createElement("script");
//     script.src = src;
//     document.body.appendChild(script);
// }

// setTimeout(() => {
//     for (const lang of languages) {
//         const script = document.createElement("script");
//         script.src = `https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/languages/${lang}.min.js`;
//         document.body.appendChild(script);
//         console.log(`Enabled hl.js for ${lang}`);
//     }

//     hljs.highlightAll();
// }, 1000);

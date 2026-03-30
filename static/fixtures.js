let currentMatchday = null;
let totalMatchdays = null;

document.addEventListener('DOMContentLoaded', () => {
    fetchFixtures();
});

async function fetchFixtures(matchday) {
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const list = document.getElementById('fixtures-list');

    loading.style.display = 'block';
    error.style.display = 'none';
    list.style.display = 'none';

    try {
        const url = matchday
            ? `/api/fixtures?matchday=${matchday}`
            : '/api/fixtures';

        const resp = await fetch(url);
        if (!resp.ok) {
            const data = await resp.json();
            throw new Error(data.error || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        currentMatchday = data.matchday;
        totalMatchdays = data.total_matchdays;

        updateNav();
        renderFixtures(data.fixtures);

        loading.style.display = 'none';
        list.style.display = 'block';

    } catch (err) {
        loading.style.display = 'none';
        error.style.display = 'block';
        error.textContent = `Failed to load fixtures: ${err.message}`;
    }
}

function changeMatchday(delta) {
    const newMd = currentMatchday + delta;
    if (newMd < 1 || newMd > totalMatchdays) return;
    fetchFixtures(newMd);
}

function updateNav() {
    document.getElementById('matchday-label').textContent = `Matchweek ${currentMatchday}`;
    document.getElementById('btn-prev').disabled = (currentMatchday <= 1);
    document.getElementById('btn-next').disabled = (currentMatchday >= totalMatchdays);
}

function renderFixtures(fixtures) {
    const list = document.getElementById('fixtures-list');
    list.innerHTML = '';

    let currentDate = '';

    fixtures.forEach(fixture => {
        const fixtureDate = new Date(fixture.date);
        const dateStr = fixtureDate.toLocaleDateString('en-GB', {
            weekday: 'long', day: 'numeric', month: 'long', year: 'numeric'
        });

        // Date header
        if (dateStr !== currentDate) {
            currentDate = dateStr;
            const header = document.createElement('div');
            header.className = 'fixture-date-header';
            header.textContent = dateStr;
            list.appendChild(header);
        }

        const card = document.createElement('div');
        card.className = 'fixture-card';

        const time = fixtureDate.toLocaleTimeString('en-GB', {
            hour: '2-digit', minute: '2-digit'
        });

        const pred = fixture.prediction;
        const isFinished = fixture.status === 'FINISHED';

        let scoreHtml = '';
        if (isFinished) {
            scoreHtml = `<div class="fixture-score">${fixture.home_goals} — ${fixture.away_goals}</div>
                         <div class="fixture-status">Full Time</div>`;
        } else {
            scoreHtml = `<div class="fixture-time">${time}</div>`;
        }

        let predictionBarHtml = '';
        let homeStatsHtml = '';
        let awayStatsHtml = '';

        if (pred) {
            const probs = [
                { label: 'H', value: pred.home_win_pct },
                { label: 'D', value: pred.draw_pct },
                { label: 'A', value: pred.away_win_pct },
            ];
            const maxProb = probs.reduce((a, b) => a.value > b.value ? a : b).label;

            predictionBarHtml = `
                <div class="prediction-bar">
                    <div class="prob-segment home-prob ${maxProb === 'H' ? 'prob-highlight' : ''}"
                         style="width: ${pred.home_win_pct}%">
                        ${pred.home_win_pct}%
                    </div>
                    <div class="prob-segment draw-prob ${maxProb === 'D' ? 'prob-highlight' : ''}"
                         style="width: ${pred.draw_pct}%">
                        ${pred.draw_pct}%
                    </div>
                    <div class="prob-segment away-prob ${maxProb === 'A' ? 'prob-highlight' : ''}"
                         style="width: ${pred.away_win_pct}%">
                        ${pred.away_win_pct}%
                    </div>
                </div>
            `;

            homeStatsHtml = `
                <div class="fixture-stat">xG: <span class="fixture-stat-value">${pred.home_xg}</span></div>
                <div class="fixture-stat">CS: <span class="fixture-stat-value">${pred.home_cs_pct}%</span></div>
            `;

            awayStatsHtml = `
                <div class="fixture-stat">xG: <span class="fixture-stat-value">${pred.away_xg}</span></div>
                <div class="fixture-stat">CS: <span class="fixture-stat-value">${pred.away_cs_pct}%</span></div>
            `;
        }

        card.innerHTML = `
            <div class="fixture-teams">
                <div class="fixture-team home-team">
                    <div class="fixture-team-info">
                        <span class="team-name">${fixture.home_team}</span>
                        ${homeStatsHtml}
                    </div>
                    <img class="team-crest" src="${fixture.home_crest}" alt="" loading="lazy">
                </div>
                <div class="fixture-centre">
                    ${scoreHtml}
                </div>
                <div class="fixture-team away-team">
                    <img class="team-crest" src="${fixture.away_crest}" alt="" loading="lazy">
                    <div class="fixture-team-info">
                        <span class="team-name">${fixture.away_team}</span>
                        ${awayStatsHtml}
                    </div>
                </div>
            </div>
            ${predictionBarHtml}
        `;

        list.appendChild(card);
    });
}

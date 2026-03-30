let euroData = [];
let currentSortCol = 'cl';
let currentSortAsc = false;
let cachedCrests = {};

document.addEventListener('DOMContentLoaded', () => {
    fetchProbabilities();
});

async function fetchProbabilities() {
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const content = document.getElementById('content');

    try {
        const [simResp, standingsResp] = await Promise.all([
            fetch('/api/season-simulation'),
            fetch('/api/standings'),
        ]);

        if (!simResp.ok) {
            const data = await simResp.json();
            throw new Error(data.error || `HTTP ${simResp.status}`);
        }

        const data = await simResp.json();
        const posProbs = data.position_probabilities;
        const projectedTable = data.projected_table;

        // Build crest lookup from standings
        const crests = {};
        if (standingsResp.ok) {
            const standingsData = await standingsResp.json();
            standingsData.teams.forEach(t => { crests[t.name] = t.crest; });
        }

        // Sort teams by projected position
        const teams = projectedTable.map(r => r.team);

        cachedCrests = crests;
        renderGrid(teams, posProbs, crests);
        buildEuroData(teams, posProbs);
        renderEuroHead();
        renderEuroTable(crests);

        loading.style.display = 'none';
        content.style.display = 'block';

    } catch (err) {
        loading.style.display = 'none';
        error.style.display = 'block';
        error.textContent = `Failed to load probabilities: ${err.message}`;
    }
}

function renderGrid(teams, posProbs, crests) {
    const grid = document.getElementById('prob-grid');
    const nPositions = teams.length;

    // Header row
    let headerHtml = '<thead><tr><th class="prob-team-col">Team</th>';
    for (let i = 1; i <= nPositions; i++) {
        headerHtml += `<th class="prob-pos-col">${i}</th>`;
    }
    headerHtml += '</tr></thead>';

    // Body rows
    let bodyHtml = '<tbody>';
    teams.forEach(team => {
        const probs = posProbs[team];
        const crestUrl = crests[team] || '';
        const crestImg = crestUrl
            ? `<img class="team-crest grid-crest" src="${crestUrl}" width="24" height="24" alt="" loading="lazy" onerror="this.style.display='none'">`
            : '';
        bodyHtml += '<tr>';
        bodyHtml += `<td class="prob-team-cell">${crestImg}${shortenName(team)}</td>`;
        probs.forEach((pct, posIdx) => {
            const rounded = Math.round(pct * 10) / 10;
            const display = rounded >= 0.5 ? rounded.toFixed(1) : '';
            const intensity = getIntensity(rounded);
            const zoneClass = getZoneClassForPos(posIdx + 1);
            bodyHtml += `<td class="prob-cell ${zoneClass}" style="background: ${intensity}">${display}</td>`;
        });
        bodyHtml += '</tr>';
    });
    bodyHtml += '</tbody>';

    grid.innerHTML = headerHtml + bodyHtml;
}

function buildEuroData(teams, posProbs) {
    euroData = teams.map(team => {
        const probs = posProbs[team];
        const cl = probs.slice(0, 5).reduce((a, b) => a + b, 0);
        const el = probs.slice(5, 7).reduce((a, b) => a + b, 0);
        const ecl = probs[7] || 0;
        const rel = probs.slice(17, 20).reduce((a, b) => a + b, 0);
        return { team, cl, el, ecl, rel };
    });
}

function renderEuroHead() {
    const thead = document.getElementById('euro-head');

    const columns = [
        { key: 'cl', label: 'Champions League' },
        { key: 'el', label: 'Europa League' },
        { key: 'ecl', label: 'Conference League' },
        { key: 'rel', label: 'Relegation' },
    ];

    let html = '<tr><th class="col-team">Team</th>';
    columns.forEach(col => {
        const isActive = currentSortCol === col.key;
        const arrow = isActive ? (currentSortAsc ? '\u25B2' : '\u25BC') : '\u25BC';
        const activeClass = isActive ? 'sort-active' : '';
        html += `<th class="col-num sortable-th ${activeClass}" data-sort="${col.key}" onclick="sortTable('${col.key}')">${col.label}<span class="sort-arrow">${arrow}</span></th>`;
    });
    html += '</tr>';

    thead.innerHTML = html;
}

function sortTable(colKey) {
    if (currentSortCol === colKey) {
        currentSortAsc = !currentSortAsc;
    } else {
        currentSortCol = colKey;
        currentSortAsc = false; // default descending
    }
    renderEuroHead();
    renderEuroTable(cachedCrests);
}

function renderEuroTable(crests) {
    crests = crests || {};
    const tbody = document.getElementById('euro-body');

    // Sort data
    const sorted = [...euroData].sort((a, b) => {
        const diff = currentSortAsc ? a[currentSortCol] - b[currentSortCol] : b[currentSortCol] - a[currentSortCol];
        return diff;
    });

    tbody.innerHTML = '';
    sorted.forEach(row => {
        // Only show teams with > 0.5% chance of any spot or relegation
        if (row.cl < 0.5 && row.el < 0.5 && row.ecl < 0.5 && row.rel < 0.5) return;

        const crestUrl = crests[row.team] || '';
        const crestImg = crestUrl
            ? `<img class="team-crest" src="${crestUrl}" width="24" height="24" alt="" loading="lazy" onerror="this.style.display='none'">`
            : '';

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>
                <div class="team-cell">${crestImg}${row.team}</div>
            </td>
            <td class="euro-cl">${row.cl.toFixed(1)}%</td>
            <td class="euro-el">${row.el.toFixed(1)}%</td>
            <td class="euro-ecl">${row.ecl.toFixed(1)}%</td>
            <td class="euro-rel">${row.rel.toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
    });
}

function shortenName(name) {
    const map = {
        'Manchester City FC': 'Man City',
        'Manchester United FC': 'Man Utd',
        'Arsenal FC': 'Arsenal',
        'Liverpool FC': 'Liverpool',
        'Chelsea FC': 'Chelsea',
        'Tottenham Hotspur FC': 'Spurs',
        'Newcastle United FC': 'Newcastle',
        'Aston Villa FC': 'Aston Villa',
        'West Ham United FC': 'West Ham',
        'Brighton & Hove Albion FC': 'Brighton',
        'Crystal Palace FC': 'C. Palace',
        'Wolverhampton Wanderers FC': 'Wolves',
        'AFC Bournemouth': 'Bournemouth',
        'Nottingham Forest FC': 'Nott. Forest',
        'Fulham FC': 'Fulham',
        'Brentford FC': 'Brentford',
        'Everton FC': 'Everton',
        'Leicester City FC': 'Leicester',
        'Leeds United FC': 'Leeds',
        'Southampton FC': 'Southampton',
        'Burnley FC': 'Burnley',
        'Sheffield United FC': 'Sheff Utd',
        'Luton Town FC': 'Luton',
        'Ipswich Town FC': 'Ipswich',
        'Sunderland AFC': 'Sunderland',
    };
    return map[name] || name;
}

function getIntensity(pct) {
    if (pct < 0.5) return 'transparent';
    const alpha = Math.min(pct / 50, 1) * 0.6 + 0.05;
    return `rgba(88, 166, 255, ${alpha})`;
}

function getZoneClassForPos(pos) {
    if (pos <= 5) return 'grid-zone-cl';
    if (pos <= 7) return 'grid-zone-el';
    if (pos === 8) return 'grid-zone-ecl';
    if (pos >= 18) return 'grid-zone-rel';
    return '';
}

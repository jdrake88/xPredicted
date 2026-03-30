document.addEventListener('DOMContentLoaded', () => {
    fetchProjectedTable();
});

async function fetchProjectedTable() {
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const wrapper = document.getElementById('table-wrapper');
    const tbody = document.getElementById('projected-body');

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
        const table = data.projected_table;

        // Build crest lookup from standings
        const crests = {};
        if (standingsResp.ok) {
            const standingsData = await standingsResp.json();
            standingsData.teams.forEach(t => { crests[t.name] = t.crest; });
        }

        tbody.innerHTML = '';
        table.forEach(row => {
            const tr = document.createElement('tr');
            tr.className = getZoneClass(row.position);

            const gdClass = row.projected_gd > 0 ? 'gd-positive' : row.projected_gd < 0 ? 'gd-negative' : '';
            const gdDisplay = row.projected_gd > 0 ? `+${row.projected_gd}` : row.projected_gd;
            const crestUrl = crests[row.team] || '';
            const crestImg = crestUrl
                ? `<img class="team-crest" src="${crestUrl}" width="24" height="24" alt="" loading="lazy" onerror="this.style.display='none'">`
                : '';

            tr.innerHTML = `
                <td>${row.position}</td>
                <td>
                    <div class="team-cell">
                        ${crestImg}
                        ${row.team}
                    </div>
                </td>
                <td>${row.current_points}</td>
                <td class="col-pts">${row.projected_points}</td>
                <td class="${gdClass}">${gdDisplay}</td>
            `;

            tbody.appendChild(tr);
        });

        loading.style.display = 'none';
        wrapper.style.display = 'block';

    } catch (err) {
        loading.style.display = 'none';
        error.style.display = 'block';
        error.textContent = `Failed to load predictions: ${err.message}`;
    }
}

function getZoneClass(position) {
    if (position <= 5) return 'zone-cl';
    if (position <= 7) return 'zone-el';
    if (position === 8) return 'zone-ecl';
    if (position >= 18) return 'zone-rel';
    return '';
}

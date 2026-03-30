document.addEventListener('DOMContentLoaded', () => {
    fetchStandings();
});

async function fetchStandings() {
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const wrapper = document.getElementById('table-wrapper');
    const seasonLabel = document.getElementById('season-label');
    const tbody = document.getElementById('standings-body');

    try {
        const resp = await fetch('/api/standings');
        if (!resp.ok) {
            const data = await resp.json();
            throw new Error(data.error || `HTTP ${resp.status}`);
        }

        const data = await resp.json();

        seasonLabel.textContent = `${data.competition} — ${data.season}`;

        tbody.innerHTML = '';
        data.teams.forEach(team => {
            const row = document.createElement('tr');
            row.className = getZoneClass(team.position);

            const gdClass = team.gd > 0 ? 'gd-positive' : team.gd < 0 ? 'gd-negative' : '';
            const gdDisplay = team.gd > 0 ? `+${team.gd}` : team.gd;

            row.innerHTML = `
                <td>${team.position}</td>
                <td>
                    <div class="team-cell">
                        <img class="team-crest" src="${team.crest}" alt="" loading="lazy">
                        ${team.name}
                    </div>
                </td>
                <td>${team.played}</td>
                <td>${team.won}</td>
                <td>${team.drawn}</td>
                <td>${team.lost}</td>
                <td class="${gdClass}">${gdDisplay}</td>
                <td class="col-pts">${team.points}</td>
            `;

            tbody.appendChild(row);
        });

        loading.style.display = 'none';
        wrapper.style.display = 'block';

    } catch (err) {
        loading.style.display = 'none';
        error.style.display = 'block';
        error.textContent = `Failed to load standings: ${err.message}`;
    }
}

function getZoneClass(position) {
    if (position <= 5) return 'zone-cl';
    if (position <= 7) return 'zone-el';
    if (position === 8) return 'zone-ecl';
    if (position >= 18) return 'zone-rel';
    return '';
}

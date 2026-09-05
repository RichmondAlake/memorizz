const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const assert = require('node:assert/strict');
const source = fs.readFileSync('packaging/npm/bin/memorizz.js', 'utf8');
const version = JSON.parse(fs.readFileSync('packaging/npm/package.json', 'utf8')).version;
for (const forwarded of [['run', '--stream', '--output', 'jsonl', 'hello'], ['run', '--no-stream', 'hello']]) {
    const calls = [];
    let exit;
    const fakeProcess = {platform: 'darwin', env: {PATH: '/fixture/bin'}, argv: ['node', 'launcher', ...forwarded],
        stderr: {write() {}}, exit(code) {exit = code;}};
    vm.runInNewContext(source, {
        __filename: '/fixture/npm/memorizz.js', process: fakeProcess,
        require(name) {
            if (name === 'child_process') return {spawnSync(exe, args, options) {
                calls.push({exe, args, options});
                if (exe === 'which') return {status: 1, stdout: ''};
                if (args[0] === '--version') return {status: 0, stdout: 'memorizz ' + version};
                return {status: 0};
            }};
            if (name === 'fs') return {realpathSync: value => value, existsSync: value => value === '/fixture/bin/memorizz'};
            if (name === 'os') return {homedir: () => '/fixture'};
            if (name === 'path') return path;
            if (name === '../package.json') return {version};
            throw Error(name);
        },
    });
    assert.equal(exit, 0);
    assert.deepEqual(Array.from(calls.at(-1).args), forwarded);
    assert.equal(calls.at(-1).options.stdio, 'inherit');
}
console.log('npm launcher forwards streaming flags, inherited stdio and exit status');

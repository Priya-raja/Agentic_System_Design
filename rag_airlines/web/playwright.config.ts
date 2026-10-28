import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./e2e',use:{baseURL:'http://127.0.0.1:3000',channel:process.env.CI?undefined:'chrome',headless:true},workers:1,timeout:60000,reporter:'list'});

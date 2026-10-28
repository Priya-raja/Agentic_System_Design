import type {Metadata} from 'next';import './globals.css';import {Shell} from '@/components/Shell';
export const metadata:Metadata={title:'AeroNova — Travel with clarity',description:'A fictional AeroNova airline assistant with auditable tools, human review and cited policy evidence.'};
export default function Layout({children}:{children:React.ReactNode}){return <html lang="en"><body><Shell>{children}</Shell></body></html>;}

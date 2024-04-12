import { Component,ElementRef,OnInit } from '@angular/core';
import gasp from 'gsap';

@Component({
  selector: 'app-banner',
  templateUrl: './banner.component.html',
  styleUrl: './banner.component.css'
})
export class BannerComponent implements OnInit{
  constructor(private element:ElementRef){}
  ngOnInit(): void {
    gasp.to(".x-50",{
      opacity:1,
      y:-10,
      stagger:1,
      ease:"power1.inOut"
    })

  }

}

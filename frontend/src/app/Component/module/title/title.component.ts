import { Component ,Input} from '@angular/core';

@Component({
  selector: 'app-title',
  templateUrl: './title.component.html',
  styleUrl: './title.component.css'
})
export class TitleComponent {
  @Input()
  title!:string;
  @Input()
  color:string="black"
  @Input()
  position:string="center"
}
